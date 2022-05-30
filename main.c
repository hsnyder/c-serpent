#include <stdio.h>
#include <stdlib.h>

#define STRINGFNS_IMPLEMENTATION
#include "stringfns.h"

#define OPTPARSE_IMPLEMENTATION
#include "optparse.h"

#include "die.h"

#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) < (b) ? (b) : (a))

int
parse_argument(char ** x, char * arg, size_t argsz)
{
	// TODO guard against arguments longer than argsz
	if(**x == '-') {

		for (unsigned i = 0; i < argsz-1 && (isalnum(**x) || **x == '_'); i++, (*x)++)
			arg[i] = **x;

		*x = skipst(*x);
		return 1;
	}
	return 0;
}

int
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

int expect_char(char ** x, char exp, const char * firstchar) 
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

void 
parse_comment(char * comment) 
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
			for (int j = 0; j < nargs; j++) 
				printf("%s ", args[j]);
			printf("\n");
		}

	}

	
}

void
parse_wrapgen_commands (char * t)
{
	char *x = t;

	while((x = strchr(x, '/'))) {
		x++;

		if(*x == '/') {
			// start single line comment
			x++;
			char * comment = strsepstr(&x,"\n");
			parse_comment(comment);
		} else if (*x == '*') {
			// start multiline comment
			x++;
			char * comment = strsepstr(&x,"*/");
			parse_comment(comment);
		}
	}
}

_Noreturn void
usage(const char * progname)
{
	const char * msg = 
		"filename\n"
		"\tfilename\tFile to operate on\n";
	die("usage: %s %s", progname, msg);
}

int 
main(int argc, char ** argv)
{
	(void) argc; 

	const char * filename = 0;
	const char * progname = argv[0];

	// parse arguments
	{
		int option; 
		struct optparse options;
		optparse_init(&options, argv);
		while ((option = optparse(&options, "p")) != -1) {
			switch(option) {
			case 'p':
				break;
			/*
			case 'm':
				if (1 != sscanf(options.optarg, "%i", &mno)) 
					die("Couldn't understand memory type number");
				break;
				*/
			case '?':
				die("%s",options.errmsg);
				break;
			}
		}
		if(!(filename = optparse_arg(&options)))
			usage(progname);
	}



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

	parse_wrapgen_commands(filedata);

	free(filedata);
}
