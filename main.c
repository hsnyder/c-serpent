#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#define MIN(a,b) ((a) > (b) ? (b) : (a))

void _Noreturn
die (const char * msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(EXIT_FAILURE);
}

void _Noreturn
pdie (const char * msg)
{
	perror(msg);
	exit(EXIT_FAILURE);
}

enum {
	RPIPE = 0,
	WPIPE = 1,
};

typedef struct {
	int r;
	int w;
} Pipe;

Pipe apipe(void)
{
	int p[2] = {};
	if(pipe(p) < 0) pdie("pipe");
	return (Pipe) {.r = p[RPIPE], .w = p[WPIPE]};
}

void closepipe(Pipe p)
{
	(void) close(p.r);
	(void) close(p.w);
}

int 
process(const char * cmd, char * const* args, int stdin_p, int stdout_p)
{
	int child = fork();
	if (0 == child) {

		if(dup2(stdin_p, STDIN_FILENO) == -1) pdie("dup2(stdin)");
		if(dup2(stdout_p, STDOUT_FILENO) == -1) pdie("dup2(stdout)");

		(void) execvp(cmd, args);

		pdie("execvp");

	} else if (child > 0) {

		return child;

	} else {
		pdie("fork");
	}
}

void * amalloc(size_t nb)
{
	void * ptr = malloc(nb);
	if(!ptr) die("malloc failed");
	return ptr;
}

#include "main_jq_query.c"

int 
main (int argc, char ** argv) 
{
	if (argc < 2) die("No target file specified");

	Pipe clang_in = apipe();
	Pipe clang_jq = apipe();
	Pipe jq_out   = apipe();

	char * clang_args[1000] = {"clang", "-c", "-o", "/dev/null" };
	{
		int i = 4 ;
		for(; i < argc; i++) clang_args[i] = argv[i-2];
		clang_args[i++] = "-Xclang"; 
		clang_args[i++] = "-ast-dump=json";
		clang_args[i++] = argv[1];
	}

	int clang = process("clang", clang_args, clang_in.r, clang_jq.w);

	char * jq_args[] = {"jq", main_jq_query_txt,NULL};

	int jq = process("jq", jq_args, clang_jq.r, jq_out.w);

	closepipe(clang_in); 
	closepipe(clang_jq);
	(void) close(jq_out.w);

	FILE *f = fdopen(jq_out.r, "r");
	if(!f) pdie("fdopen(jq_stdout)");

	char line [1<<20];
	while(fgets(line,sizeof(line),f)) fputs(line,stdout);

	fclose(f);

}
