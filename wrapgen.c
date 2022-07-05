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

#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) < (b) ? (b) : (a))

#include "preamble.c"
#include "example.c"

#include "pdjson.c"

static inline int 
is_alnum_uscore(char x) 
{
	return isalnum(x) || x == '_';
}

static int
parse_argument(char ** x, char * arg, size_t argsz)
{
	// TODO guard against arguments longer than argsz
	if(**x == '-') {
		unsigned i = 0;
		do {
			arg[i] = **x;
			(*x)++;
			i++;
		} while (i < argsz && is_alnum_uscore(**x));

		*x = skipst(*x);
		return 1;
	}
	return 0;
}

static int
parse_fnname(char ** x, char * fn, size_t fnsz)
{
	// TODO guard against function names larger than fnsz
	unsigned i = 0;
	for (i = 0; i < fnsz-1 && (isalnum(**x) || **x == '_'); i++, (*x)++)
		fn[i] = **x;

	*x = skipst(*x);

	if (**x == ',') {
		(*x)++;
		*x = skipst(*x);
		return 1;
	}

	return 0;
}

static int 
expect_char(char ** x, char exp, const char * firstchar) 
{
	if (**x != exp) {
		char context[80] = {};
		snprintf (context, sizeof(context), "%s", MAX(firstchar, (*x - 40)) );
		warn("expected '%c', skipping this candidate WRAPGEN directive\n%s\n", exp, context);
		(*x)++;
		return 0;
	}
	(*x)++;
	return 1;
}

static void 
parse_comment(char * comment, int parse_args) 
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

		x += len;
		x = skipst(x);
		char args[10][80] = {};
		int nargs = 0;
		while (nargs < 10 && parse_argument(&x, args[nargs++], 80));

		if(!expect_char(&x, '(', comment)) continue;

		char fns[100][80] = {};
		int nfns = 0;
		while (nfns < 100 && parse_fnname(&x, fns[nfns++], 80));

		if(!expect_char(&x, ')', comment)) continue;
	
		for (int i = 0; i < nfns; i++) {
			printf("%s ", fns[i]);
			if (parse_args)
				for (int j = 0; j < nargs; j++) 
					printf("%s ", args[j]);
			printf("\n");
		}
	}
}

static void
parse_wrapgen_commands (const char * filename, int parse_args)
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
			parse_comment(comment, parse_args);
		} else if (*x == '*') {
			// start multiline comment
			x++;
			char * comment = strsepstr(&x,"*/");
			parse_comment(comment, parse_args);
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
usage(const char * progname)
{
	const char * msg = 
		"[-cnpe] filename [extra...]\n"
		"\t-c      \tParse wrapgen commands from souce file and then exit.\n"
		"\t-n      \tDon't parse wrapgen command arguments (meaningful only with -c)\n"
		"\t-p      \tEmit preamble (helper macros required for the generated wrappers) and then exit.\n"
		"\t-e      \tEmit example module definition and then exit.\n"
		"\tfilename\tFile to operate on.\n"
		"\textra   \tExtra flags to pass to clang.\n";
	die("usage: %s %s", progname, msg);
}

enum mode {
	FORK = 0,
	PARSE_COMMANDS,
	PREAMBLE,
	EXAMPLE,
};

int 
main(int argc, char ** argv)
{
	const char * progname = argv[0];
	const char * filename = 0;
	int narg = 0;

	enum mode m = FORK;
	int parse_args = 1;

	const char * extra_args[100] = {};
	if (argc > 100) die("wrapgen doesn't support more than 100 command line arguments");

	// parse arguments
	{
		int option; 
		struct optparse options;
		optparse_init(&options, argv);
		options.permute = 0;
		while ((option = optparse(&options, "cpne")) != -1) {
			switch(option) {
			case 'c':
				m = PARSE_COMMANDS;
				break;
			case 'p':
				m = PREAMBLE;
				break;
			case 'e':
				m = EXAMPLE;
				break;
			case 'n':
				parse_args = 0;
				break;
			case '?':
				die("%s",options.errmsg);
				break;
			}
		}

		if(!(filename = optparse_arg(&options)) 
				&& m != PREAMBLE
				&& m != EXAMPLE )
			usage(progname);

		while((extra_args[narg] = optparse_arg(&options))) 
			narg++;
	}

	switch (m) {
	
	case FORK:
		1==1;
		char *cmd_bufr[130] = {"clang", "-c", "-o", "/dev/null"};
		int x = 4;
		for (int i = 0; i < narg; i++) cmd_bufr[x++] = extra_args[i];
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

		break;

	case PARSE_COMMANDS:
		parse_wrapgen_commands(filename, parse_args);
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
