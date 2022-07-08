#include <stdio.h>
#include <stdlib.h>

#define STRINGFNS_IMPLEMENTATION
#include "stringfns.h"

#define OPTPARSE_IMPLEMENTATION
#include "optparse.h"

#include "die.h"
#define expect(cond) do{if(!(cond))die("expectation violation: %s", #cond);}while(0)
#define expect2(cond, ...) do{if(!(cond))die(__VA_ARGS__);}while(0)

#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) < (b) ? (b) : (a))

#include "preamble.c"
#include "example.c"

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

int 
main (int argc, char ** argv)
{
	expect2 (argc>1, "No filename supplied");

	FILE * f = fopen(argv[1],"r");
	expect2 (f, "Coudln't open %s", argv[1]);

	static char filecontent[1<<25] = {0};
	size_t n = fread (filecontent,sizeof(filecontent)-1, f);
	expect2 (n < sizeof(filecontent)-1, "You really have a code file that's more than 32 MiB? o.O");

	fclose(f);

	
	

	switch (m) {
	
	case FORK:
		die("fork not implemented");
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

	case FUNCTION;
		wrapgen_fn();
		break;

	default:
		die("bug: invalid mode: %i", (int) m);
		break;
	} 
}


