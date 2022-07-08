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

struct options {
	int error_arg;
};

const struct options DEFAULT_OPTIONS = {.error_arg = -1};


struct task {
	char *fn_name;
	struct options opts;
};


#define MAX_TASKS 1000
static struct task tasks[MAX_TASKS] = {0};
static int n_tasks = 0;

static void 
parse_single_command(char * s) 
{
	static char *tokens[1<<20] = {0};
	int n = 0;

	char *y = 0;
	char *x = skipwhitespace(s);

	while ((y = strtok(x," \t"))) {

		if (strlen(y)) tokens[n++] = y;	
		expect(n < (1<<20));
		x = 0;
	}

	struct options o = DEFAULT_OPTIONS;

	int i = 0;
	for (i = 0; i < n; i++) {
		if (tokens[i][0] == '-') {
			
			switch (tokens[i][1]) {
			case 'e':
				o.error_arg = atoi(tokens[i]+2);
				break;
			default:
				die("Invalid wrapgen flag %s", tokens[i]);
			}
		}
		else break;
	}

	for (; i < n; i++) {
		struct task t = {
			.fn_name = tokens[i],
			.opts = o,
		};

		tasks[n_tasks++] = t;
		expect(n_tasks < MAX_TASKS);
	}
}

static void
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

		// find the "length" of the occurrence
		size_t spn = strcspn(x, "\r\n*/"); // technically this means that * and / characters seperately terminate the command, but that's ok
		x[spn] = 0;
		parse_single_command(x+len);

		x += spn+1;
	}
}


static void
parse_wrapgen_commands (const char * x)
{
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

static int
is_on_todo_list(const char * s, task * t)
{
	for (int i = 0; i < n_tasks; i++) {
		if(!strncmp(tasks[i].fn_name, s, strlen(tasks[i].fn_name))) {
			*t = tasks[i];
			return 1;
		}
	}
	return 0;
}

static void
emit_wrapper (task t, char * sig) 
{
	// parse the function signature...
	char *args[100] = {0};
	int narg = 0;
	char *retval = strsepstr(&sig, "(");
	sig = remove_substr(sig,")");
	sig = remove_substr(sig,"const");
	sig = remove_substr(sig,"restrict");
	sig = remove_substr(sig,"volatile");

	char * tok = 0;
	while((tok = strsepstr(sig, ","))) {
		nargs[narg++] = tok;
		expect(narg < 100);
	}

	for(int i = 0; i < narg; i++) {
		remove_substr(args[i], "const");
		remove_substr(args[i], "restrict");
		remove_substr(args[i], "volatile");
	}



}

int 
main (int argc, char ** argv)
{
	expect2 (argc>1, "No filename supplied");

	FILE * f = fopen(argv[1],"r");
	expect2 (f, "Coudln't open %s", argv[1]);

	static char filecontent[1<<25] = {0};
	size_t n = fread (filecontent,1,sizeof(filecontent)-1, f);
	expect2 (n < sizeof(filecontent)-1, "You really have a code file that's more than 32 MiB? o.O");

	fclose(f);

	parse_wrapgen_commands (filecontent);	

	while (1) {
		char fn  [4096]  = {0};
		char sig [4096] = {0};

		int ok = 
			fgets (fn,  sizeof(fn),  stdin) && 
			fgets (sig, sizeof(sig), stdin);

		if (!ok) break;

		task t;
		if (is_on_todo_list(fn, &t)) {
			emit_wrapper(t,sig);
		}
	}
}


