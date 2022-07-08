#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>

#define STRINGFNS_IMPLEMENTATION
#include "stringfns.h"

#define OPTPARSE_IMPLEMENTATION
#include "optparse.h"

#include "die.h"

#define expect(cond) _expect(cond, #cond)
static void _expect(int cond, const char * str) {
	if(!cond) die("exectation violation: %s", %cond);
}

#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) < (b) ? (b) : (a))

#include "preamble.c"
#include "example.c"

#include "pdjson.c"
#include "buf.h"

/*
struct {

	size_t sz;
	size_t used;
	char   *heap;

} stringpile = {};

static uint64_t 
newstr (char * s) {

	size_t l = strlen(s);
	while (stringpile.used + l >= stringpile.sz) {
		stringpile.sz = MAX(1<<23, 2*stringpile.sz);
		stringpile.heap = realloc(stringpile.heap, stringpile.sz);
		if (!stringpile.heap) die("out of memory");
	}

	uint64_t s = stringpile.used | (l << 44);
	return s;
}

static size_t 
len (uint64_t s) {
	return s >> 44;
}

static size_t
idx (uint64_t s) {
	return s & (~(0x00ULL) >> 20);
}


typedef struct {

	int error_arg; // -1 means retval, 0 means none, 1-based ordinal means n-th arg

} wrapgen_opts;

typedef struct {
} ; 
*/

char * progname = 0;
char ** commands = 0;

static void 
parse_comment(char * comment, int print_output) 
{
	const char * key = "WRAPGEN";
	const size_t len = strlen(key);
	char * x = comment;

	while ((x = strstr(x, key))) {

		// first check if this is a "real" occurrence
		// i.e. the characters on either side are not alphanumeric
		if (x > comment && isalnum(*(x-1))) {
			x += len;
			continue;
		}
		if (isalnum(*(x+len))) {
			x += len;
			continue;
		}

		// find the "length" of the occurrence
		size_t spn = strcspn(x, "\r\n*/"); // technically this means that * and / characters seperately terminate the command, but that's ok
		char buf[2048] = {};
		char buf2[4096] = {};
		memcpy(buf,x + len,MIN(spn,sizeof(buf)-1));
		snprintf(buf2,sizeof(buf2),"%s -X %s", progname, buf);

		if (print_output) 
			puts(buf2);
		buf_push(commands, strdup(buf2));

		x += spn;
	}
}

static void
parse_wrapgen_commands (const char * filename, int print_output)
{
	char * filedata = 0;
	// read specified file. 
	{
		FILE * f = fopen(filename, "rb");
		if(!f) die("couldn't open %s", filename);
		if(fseek(f, 0, SEEK_END))
			die("couldn't seek to end of %s", filename);
		const unsigned long sz = ftell(f);
		rewind(f);
		filedata = malloc(sz);
		if(!filedata)
			die("couldn't allocate %li bytes", sz);
		if(sz != fread(filedata, 1, sz, f))
			die("couldn't read entirety of %s", filename);
		fclose(f);
	}

	char *x = filedata;

	while((x = strchr(x, '/'))) {
		x++;

		if(*x == '/') {
			// start single line comment
			x++;
			char * comment = strsepstr(&x,"\n");
			parse_comment(comment, print_output);
		} else if (*x == '*') {
			// start multiline comment
			x++;
			char * comment = strsepstr(&x,"*/");
			parse_comment(comment, print_output);
		}
	}

	free(filedata);
}



void indent(int n)
{
	for (int i = 0; i < n * 2; i++)
		putchar(' ');
}

void pretty(json_stream *json);

void pretty_array(json_stream *json)
{
	printf("[\n");
	int first = 1;
	while (json_peek(json) != JSON_ARRAY_END && !json_get_error(json)) {
		if (!first)
			printf(",\n");
		else
			first = 0;
		indent(json_get_depth(json));
		pretty(json);
	}
	json_next(json);
	printf("\n");
	indent(json_get_depth(json));
	printf("]");
}

void pretty_object(json_stream *json)
{
	printf("{\n");
	int first = 1;
	while (json_peek(json) != JSON_OBJECT_END && !json_get_error(json)) {
		if (!first)
			printf(",\n");
		else
			first = 0;
		indent(json_get_depth(json));
		json_next(json);
		printf("\"%s\": ", json_get_string(json, NULL));
		pretty(json);
	}
	json_next(json);
	printf("\n");
	indent(json_get_depth(json));
	printf("}");
}

void pretty(json_stream *json)
{
	enum json_type type = json_next(json);
	switch (type) {
		case JSON_DONE:
			return;
		case JSON_NULL:
			printf("null");
			break;
		case JSON_TRUE:
			printf("true");
			break;
		case JSON_FALSE:
			printf("false");
			break;
		case JSON_NUMBER:
			printf("%s", json_get_string(json, NULL));
			break;
		case JSON_STRING:
			printf("\"%s\"", json_get_string(json, NULL));
			break;
		case JSON_ARRAY:
			pretty_array(json);
			break;
		case JSON_OBJECT:
			pretty_object(json);
			break;
		case JSON_OBJECT_END:
		case JSON_ARRAY_END:
			return;
		case JSON_ERROR:
			fprintf(stderr, "error: %zu: %s\n",
					json_get_lineno(json),
					json_get_error(json));
			exit(EXIT_FAILURE);
	}
}


static int
spawn (int in, int out, char *c[])
{
	pid_t p = fork(); 
	
	if (!p) {

		// child process
		if (in != 0) {
			if (dup2(in, 0) < 0) die("dup2 failed on stdin");
			close(in);
		}

		if (out != 1) {
			if (dup2(out,1) < 0) die("dup2 failed on stdout");
			close(out);
		} 

		execvp(c[0], c);

		die("exec failed");
		
	} else if (p < 0) die("fork failed");

	return p;
}


static _Noreturn void
usage(void)
{
	const char * msg = 
		"[-cnpe] filename [extra...]\n"
		"\t-X      \tWrap function (internal use, don't use this directly.\n"
		"\t-P      \tEmit preamble (helper macros required for the generated wrappers) and then exit.\n"
		"\t-E      \tEmit example module definition and then exit.\n"
		"\t-C      \tParse commands and exit (for debugging wrapgen.)\n"
		"\tfilename\tFile to operate on.\n"
		"\textra   \tExtra flags to pass to clang.\n";
	die("usage: %s %s", progname, msg);
}

enum mode {
	NORMAL = 0,
	PARSE_COMMANDS,
	PREAMBLE,
	EXAMPLE,
	WRAPFN,
};

int 
main(int argc, char ** argv)
{
	progname = argv[0];
	const char * filename = 0;
	int narg = 0;

	enum mode m = NORMAL;

	const char * extra_args[100] = {};
	if (argc > 100) die("wrapgen doesn't support more than 100 command line arguments");

	// parse arguments
	{
		int option; 
		struct optparse options;
		optparse_init(&options, argv);
		options.permute = 0;
		while ((option = optparse(&options, "XPEC")) != -1) {
			switch(option) {
			case 'C':
				m = PARSE_COMMANDS;
				break;
			case 'X':
				m = WRAPFN;
				break;
			case 'P':
				m = PREAMBLE;
				break;
			case 'E':
				m = EXAMPLE;
				break;
			case '?':
				die("%s",options.errmsg);
				break;
			}
		}

		while((extra_args[narg] = optparse_arg(&options))) 
			narg++;

		filename = extra_args[0];
		if (narg == 0 && !(m == PREAMBLE || m == EXAMPLE)) usage();
	}

	switch (m) {
	
	case NORMAL:
		parse_wrapgen_commands(filename, 0);

		char *cmd_bufr[130] = {"clang", "-c", "-o", "/dev/null"};
		int x = 4;
		for (int i = 1; i < narg; i++) cmd_bufr[x++] = extra_args[i];
		cmd_bufr[x++] = "-Xclang";
		cmd_bufr[x++] = "-ast-dump=json";
		cmd_bufr[x++] = filename;

		
		int pi[2] = {-1,-1};
		int po[2] = {-1,-1};

		pipe(pi);
		pipe(po);

		pid_t pid = spawn(pi[0],po[1],cmd_bufr);

		close(pi[1]);
		close(po[1]);

		json_stream js = {};
		FILE * f = fdopen(po[0], "rb");
		json_open_stream(&js, f);
		json_set_streaming(&js,false);
		pretty(&js);
		if(json_get_error(&js)) {
			die("error %zu %s\n", json_get_lineno(&js), json_get_error(&js));
		}
		json_close(&js);
		fclose(f);
		close(po[0]);

		int status = 0;
		waitpid(pid, &status, 0);

		/*
		for (int i = 0 ; i < buf_size(commands); i++) {
			printf("%s\n", commands[i]);
		}
		*/

		break;

	case PARSE_COMMANDS:
		parse_wrapgen_commands(filename, 1);
		break;

	case PREAMBLE:
		for (unsigned i = 0; i < preamble_len; i++) 
			fputc(preamble[i], stdout);
		break;

	case EXAMPLE:
		for (unsigned i = 0; i < example_len; i++) 
			fputc(example[i], stdout);
		break;
	} 
}
