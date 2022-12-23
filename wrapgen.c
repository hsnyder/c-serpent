#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <stdint.h>
#include <ctype.h>

#define STB_C_LEXER_IMPLEMENTATION
#include "stb_c_lexer.h"
#include "buf.h"


/*
	==========================================================
		Helper functions
	==========================================================
*/

_Noreturn void
die (const char * fmt, ...)
{
	va_list va;
	va_start(va, fmt);
	vfprintf(stderr, fmt, va);
	va_end(va);
	fprintf(stderr, "\n");
	exit(EXIT_FAILURE);
}

#define expect(cond, ...) if(!(cond)){die(__VA_ARGS__ );}

int 
xatoi (const char *x, int *nchars_read)
{
	const char * save = x;

	int sign = 1;
	int n = 0;
	int v = 0;
	
	if(x[0] == '-') {sign = -1; x++;}
	if(x[0] == '+') {sign =  1; x++;}

	while (x[0]  &&  x[0] >= 48  &&  x[0] <= 48+9)
	{
		int digit = x[0] - 48;

		if (INT_MAX / 10 < v) goto overflow;
		v *= 10;

		if (INT_MAX - digit < v) goto overflow;
		v += digit;

		n++;
		x++;
	}
	
	if (!n) die("couldn't parse '%6s' as an integer", save);

	if (nchars_read) *nchars_read = x-save;
	return v * sign;

overflow:
	die("integer overflow when trying to convert '%14s'", save);
}

typedef struct {
	struct {
		int present;
		int argno;
		const char * fn;
	} err;
} wrapgen_options;



/*
	==========================================================
		C99 parsing
	==========================================================
*/

void gen(const char * s)
{
	abort();
}

int eat(stb_lexer *lex, int token, const char *keyword)
{
	if(lex->token == token) {
		if(token == CLEX_id) {
			assert(keyword);
			if (strcmp(keyword, lex->string) goto error;
		}
		return stb_c_lexer_get_token(lex);
	}

	char saw[100] = {0};
	char wanted[100] = {0};
	error: switch (lex->token) {
		case CLEX_id        : snprintf(saw, sizeof(saw), "identifier %s", lex->string); break;
		case CLEX_eq        : snprintf(saw, sizeof(saw), "=="); break;
		case CLEX_noteq     : snprintf(saw, sizeof(saw), "!="); break;
		case CLEX_lesseq    : snprintf(saw, sizeof(saw), "<="); break;
		case CLEX_greatereq : snprintf(saw, sizeof(saw), ">="); break;
		case CLEX_andand    : snprintf(saw, sizeof(saw), "&&"); break;
		case CLEX_oror      : snprintf(saw, sizeof(saw), "||"); break;
		case CLEX_shl       : snprintf(saw, sizeof(saw), "<<"); break;
		case CLEX_shr       : snprintf(saw, sizeof(saw), ">>"); break;
		case CLEX_plusplus  : snprintf(saw, sizeof(saw), "++"); break;
		case CLEX_minusminus: snprintf(saw, sizeof(saw), "--"); break;
		case CLEX_arrow     : snprintf(saw, sizeof(saw), "->"); break;
		case CLEX_andeq     : snprintf(saw, sizeof(saw), "&="); break;
		case CLEX_oreq      : snprintf(saw, sizeof(saw), "|="); break;
		case CLEX_xoreq     : snprintf(saw, sizeof(saw), "^="); break;
		case CLEX_pluseq    : snprintf(saw, sizeof(saw), "+="); break;
		case CLEX_minuseq   : snprintf(saw, sizeof(saw), "-="); break;
		case CLEX_muleq     : snprintf(saw, sizeof(saw), "*="); break;
		case CLEX_diveq     : snprintf(saw, sizeof(saw), "/="); break;
		case CLEX_modeq     : snprintf(saw, sizeof(saw), "%%="); break;
		case CLEX_shleq     : snprintf(saw, sizeof(saw), "<<="); break;
		case CLEX_shreq     : snprintf(saw, sizeof(saw), ">>="); break;
		case CLEX_eqarrow   : snprintf(saw, sizeof(saw), "=>"); break;
		case CLEX_dqstring  : snprintf(saw, sizeof(saw), "double-quoted string \"%s\"", lex->string); break;
		case CLEX_sqstring  : snprintf(saw, sizeof(saw), "single-quoted string '%s'", lex->string); break;
		case CLEX_charlit   : snprintf(saw, sizeof(saw), "character literal '%s'", lex->string); break;
		case CLEX_intlit    : snprintf(saw, sizeof(saw), "integer literal %ld", lex->int_number); break;
		case CLEX_floatlit  : snprintf(saw, sizeof(saw), "floating point literal %g", lex->real_number); break;
		default:
		      if (lex->token >= 0 && lex->token < 256)
			      snprintf(saw, sizeof(saw), "%c", (int) lex->token);
		      else {
			      snprintf(saw, sizeof(saw), "<<<UNKNOWN TOKEN %ld >>>\n", lex->token);
			      die("");
		      }
		      break;
	}

	switch (token) {
		case CLEX_id: 
			if (keyword) 
				snprintf(wanted, sizeof(wanted), "keyword %s", keyword);
			else
				snprintf(wanted, sizeof(wanted), "identifier");
			break;
		case CLEX_eq        : snprintf(wanted, sizeof(wanted), "=="); break;
		case CLEX_noteq     : snprintf(wanted, sizeof(wanted), "!="); break;
		case CLEX_lesseq    : snprintf(wanted, sizeof(wanted), "<="); break;
		case CLEX_greatereq : snprintf(wanted, sizeof(wanted), ">="); break;
		case CLEX_andand    : snprintf(wanted, sizeof(wanted), "&&"); break;
		case CLEX_oror      : snprintf(wanted, sizeof(wanted), "||"); break;
		case CLEX_shl       : snprintf(wanted, sizeof(wanted), "<<"); break;
		case CLEX_shr       : snprintf(wanted, sizeof(wanted), ">>"); break;
		case CLEX_plusplus  : snprintf(wanted, sizeof(wanted), "++"); break;
		case CLEX_minusminus: snprintf(wanted, sizeof(wanted), "--"); break;
		case CLEX_arrow     : snprintf(wanted, sizeof(wanted), "->"); break;
		case CLEX_andeq     : snprintf(wanted, sizeof(wanted), "&="); break;
		case CLEX_oreq      : snprintf(wanted, sizeof(wanted), "|="); break;
		case CLEX_xoreq     : snprintf(wanted, sizeof(wanted), "^="); break;
		case CLEX_pluseq    : snprintf(wanted, sizeof(wanted), "+="); break;
		case CLEX_minuseq   : snprintf(wanted, sizeof(wanted), "-="); break;
		case CLEX_muleq     : snprintf(wanted, sizeof(wanted), "*="); break;
		case CLEX_diveq     : snprintf(wanted, sizeof(wanted), "/="); break;
		case CLEX_modeq     : snprintf(wanted, sizeof(wanted), "%%="); break;
		case CLEX_shleq     : snprintf(wanted, sizeof(wanted), "<<="); break;
		case CLEX_shreq     : snprintf(wanted, sizeof(wanted), ">>="); break;
		case CLEX_eqarrow   : snprintf(wanted, sizeof(wanted), "=>"); break;
		case CLEX_dqstring  : snprintf(wanted, sizeof(wanted), "double-quoted string"); break;
		case CLEX_sqstring  : snprintf(wanted, sizeof(wanted), "single-quoted string"); break;
		case CLEX_charlit   : snprintf(wanted, sizeof(wanted), "character literal"); break;
		case CLEX_intlit    : snprintf(wanted, sizeof(wanted), "integer literal"); break;
		case CLEX_floatlit  : snprintf(wanted, sizeof(wanted), "floating point literal"); break;
		default:
		      if (lex->token >= 0 && lex->token < 256)
			      snprintf(wanted, sizeof(wanted), "%c", (int) lex->token);
		      else {
			      die("invalid argument to 'eat'");
		      }
		      break;
	}
	
	stb_lex_location loc = {0};
	stb_c_lexer_get_location(lex, lex->where_firstchar, &loc);
	die("Parse error at line %i, character %i: Expected %s, found %s", loc->line_number, loc->line_offset, wanted, saw);
	return 0;
}


void external_declaration(stb_lexer *lex)
{
	
}


void translation_unit(stb_lexer *lex)
{
	while (stb_c_lexer_get_token(lex)) {
		external_declaration(lex);
	}
}


/*
	==========================================================
		Comment parsing, main, etc
	==========================================================
*/


void 
wrap_identifier (const char * filetext, const char * identifier, wrapgen_options opts)
{
	size_t file_len = strlen(filetext);
	char * scratch = malloc(file_len+1);
	if(!scratch) die("out of memory");

	stb_lexer lex;
	stb_c_lexer_init(&lex, filetext, filetext+file_len, scratch, file_len+1);

	while (stb_c_lexer_get_token(&lex)) {
		if (lex.token == CLEX_parse_error) {
			die("<<<PARSE ERROR>>>");
			break;
		}

		// continue here
	}

	free (scratch);
}

void 
main_x (const char *filetext, char *argv[])
{
	/*
		Wrapgen flags are:
		-e,arg,chkfn add custom error checking (call chkfn and pass arg (specified by number))
	*/

	wrapgen_options options = {0};

	while (argv[0]) 
	{
		const char *a = argv[0];

		if (a[0]=='-' && a[1]=='e') {

			const char * errmsg = "malformed -e flag (format is -e,argno,functiontocall)";
			expect(a[2]==',', errmsg);

			int n = 0;
			options.err.argno = xatoi(a+3, &n);

			expect(a[n]==',', errmsg); 
			options.err.fn = a+n+1; 

			options.err.present = 1;
		}

		else if (a[0] == '-') {
			die("wrapgen flag '%s' not understood", a);
		}

		else {
			wrap_identifier(filetext, a, options);
		}
			
		argv++;
	}
}

void 
execute_command (const char *filetext, const char *cmd, long sz)
{
	/* 
		Tokenize the provided command and pass the argument list to main_x
	*/
	
	char buf[4096] = {0};
	if (sz >= sizeof(buf)) 
		die("Wrapgen command lines must be < 4kB");
	memcpy(buf,cmd,sz);

	char **args = 0;

	long argstart = -1;
	for (long i = 0; i < sz; i++) 
	{
		if (argstart < 0  &&  !isspace(buf[i]))
			argstart = i;

		else if (argstart >= 0  &&  isspace(buf[i])) {
			buf[i] = 0;
			buf_push(args, buf+argstart);
			argstart = -1;
		}
	}

	main_x(filetext, args);
	buf_free(args);
}

const char * 
command_until_eol (const char *filetext, const char *c)
{
	/* 
		Assume that from c until the end of the line (or file) is a wrapgen command and run it. 
		Returns either the null terminator, or the first char past the end of the line, as applicable. 
	*/

	const char * end = c;
	const char * retn = 0;
	for(;;)
	{
		if(*end == 0) {
			retn = end;
			break;
		}
		if(*end == '\n') {
			retn = end + 1;
			break;
		}

		end++;
	}

	execute_command(filetext, c, end-c);

	return retn;
}

const char *  
process_slashslash_comment (const char *filetext, const char *c)
{
	/*
		Assume that a single-line comment starts at c, and parse it for wrapgen commands.
		Run any wrapgen commands encountered.
		Returns either the null terminator, or the first char past the end of the comment, as applicable. 
	*/
	int search = 1;

	for (;;) {
		if (c[0] == 0) return c;
		if (c[0] == '\n') return c+1;

		if (search) 
		{
			if (*c == 'w'  &&  !strncmp(c,"wrapgen",7)  &&  isspace(c[7])) 
			{
				return command_until_eol(filetext, c+7);
			} 
			else if (!isspace(*c)) 
			{
				search = 0;
			}
		}

		c++;
	}
}

const char *  
process_slashstar_comment (const char *filetext, const char *c)
{
	/*
		Assume that a multi-line comment starts at c, and parse it for wrapgen commands.
		Run any wrapgen commands encountered.
		Returns either the null terminator, or the first char past the end of the comment, as applicable. 
	*/
	int search = 1;

	for(;;) {
		if (c[0] == 0) return c;
		if (c[0] == '*'  &&  c[1] == '/') return c+2;

		if (search)
		{
			if (*c == 'w'  &&  !strncmp(c,"wrapgen",7)  &&  isspace(c[7])) 
			{
				c = command_until_eol(filetext, c+7);
				search = 1;
			} 
			else if (!isspace(*c)) 
			{
				search = 0;
			}
		}

		if (c[0] == '\n') {
			search = 1;
		}

		c++;
	}
}

void 
usage (char *argv[]) 
{ 
	die( "usage: %s filename [filenames...]\n" , argv[0]); 
}

int 
main (int argc, char *argv[])
{
	if (!argv[1])
		usage(argv);
	argv++;


	long bufsz = 1<<20;
	char * buf = malloc(bufsz);
	if(!buf) die("out of memory");

	while(argv[0])
	{
		FILE * f = 0; 
		if (!(f = fopen(argv[0], "rb")))
			die("couldn't open %s", argv[0]);

		if (fseek(f, 0, SEEK_END))
			die("couldn't fseek on %s", argv[0]);

		long fsz = ftell(f);
		rewind(f);
		if (fsz == -1L)
			die("couldn't ftell on %s", argv[0]);

		if (fsz+1 > bufsz) {
			buf = realloc(buf, fsz+1);
			if (!buf) die("out of memory");
		}
		
		if (fsz != fread (buf, 1, fsz, f))
			die ("size of %s changed during run", argv[0]);

		buf[fsz] = 0;
		
		const char *c = buf;
		while(*c) 
		{
			if (c[0] == '/'  &&  c[1] == '/') 
				c = process_slashslash_comment(buf, c+2);

			else if (c[0] == '/'  &&  c[1] == '*') 
				c = process_slashstar_comment(buf, c+2);

			else c++;
		}

		fclose(f);
		argv++;
	}
	
	exit(EXIT_SUCCESS);
}
