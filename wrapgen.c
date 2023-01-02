#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <stdint.h>
#include <ctype.h>
#include <stdint.h>
#include <limits.h>


#define STB_C_LEXER_IMPLEMENTATION
#include "stb_c_lexer.h"
#include "buf.h"


/*
	==========================================================
		Helper functions
	==========================================================
*/

#define COUNT_ARRAY(x) ((int64_t)(sizeof(x)/sizeof(x[0])))

#define OPTIONAL(x) ((x), 1)
#define RESET()  (*p = p_saved);

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


void nop(void) {}

struct ht {
	char **ht;
	int32_t len;
	int32_t exp;
};

uint64_t hash(char *s, int32_t len)
{
	uint64_t h = 0x100;
	for (int32_t i = 0; i < len; i++) {
		h ^= s[i] & 255;
		h *= 1111111111111111111;
	}
	return h ^ h>>32;
}

int32_t ht_lookup(uint64_t hash, int exp, int32_t idx)
{
	uint32_t mask = ((uint32_t)1 << exp) - 1;
	uint32_t step = (hash >> (64 - exp)) | 1;
	return (idx + step) & mask;
}

char *strdup_len_or_die(char * str, int len)
{
	char *x = malloc(len+1);
	if(!x) die("strdup_len_or_die: out of memory");
	memcpy(x,str,len+1);
	return x;
}

char *intern(struct ht *t, char *key, int keylen)
{
	uint64_t h = hash(key, keylen+1);
	for (int32_t i = h;;) {
		i = ht_lookup(h, t->exp, i);
		if (!t->ht[i]) {
			// empty, insert here
			if ((uint32_t)t->len+1 == (uint32_t)1<<t->exp) {
				die("out of memory in intern");
			}
			t->len++;
			t->ht[i] = strdup_len_or_die(key, keylen);
			return t->ht[i];
		} else if (!strcmp(t->ht[i], key)) {
			// found, return canonical instance
			return t->ht[i];
		}
	}
}



typedef struct {
	long toktype; // this will be one of the enum values in stb_c_lexer
	int string_len;
	union {
		double real_number;
		long long int_number;
		char * string;
	};
} token;



/*
	==========================================================
		C99 parsing
	==========================================================
*/


typedef struct {
	token *tokens;   
	token *tokens_end;
} parse_ctx;

int eat_identifier(parse_ctx *p, const char *id)
{
	if(p->tokens == p->tokens_end) return 0;
	if(p->tokens[0].toktype != CLEX_id) return 0;
	if(!strlen(id) == p->tokens[0].string_len) return 0;
	if(!memcmp(id, p->tokens[0].string, p->tokens[0].string_len)) {
		p->tokens++;
		return 1;
	}
	return 0;
}

int eat_token(parse_ctx *p, long toktype)
{
	if(p->tokens == p->tokens_end) return 0;
	if(p->tokens[0].toktype == toktype) {
		p->tokens++;
		return 1;
	}
	return 0;
}


// -----------------------------------------------------------------------


int external_declaration(parse_ctx *p);
int function_definition(parse_ctx *p);
int declaration(parse_ctx *p);
int declaration_specifiers(parse_ctx *p);
int declaration_specifier(parse_ctx *p);
int declarator(parse_ctx *p);
int declaration_list(parse_ctx *p);
int compound_statement(parse_ctx *p);
int declaration_or_statement(parse_ctx *p);
int init_declarator_list(parse_ctx *p);
int init_declarator(parse_ctx *p);
int static_assert_declaration(parse_ctx *p);
int storage_class_specifier(parse_ctx *p);
int type_specifier(parse_ctx *p);
int typedef_name(parse_ctx *p);
int type_qualifier(parse_ctx *p);
int function_specifier(parse_ctx *p);
int alignment_specifier(parse_ctx *p);
int pointer(parse_ctx *p);
int direct_declarator(parse_ctx *p);
int identifier_list(parse_ctx *p);
int initializer_list(parse_ctx *p);
int designative_initializer(parse_ctx *p);
int initializer(parse_ctx *p);
int constant_expression(parse_ctx *p);
int atomic_type_specifier(parse_ctx *p);
int struct_or_union_specifier(parse_ctx *p);
int struct_or_union(parse_ctx *p);
int struct_declaration_list(parse_ctx *p);
int struct_declaration(parse_ctx *p);
int enum_specifier(parse_ctx *p);
int enumerator_list(parse_ctx *p);
int enumerator(parse_ctx *p);
int enumeration_constant(parse_ctx *p);
int type_name(parse_ctx *p);
int specifier_qualifier_list(parse_ctx *p);
int specifier_qualifier(parse_ctx *p);
int abstract_declarator(parse_ctx *p);
int direct_abstract_declarator(parse_ctx *p);
int type_qualifier_list(parse_ctx *p);
int parameter_type_list(parse_ctx *p);
int struct_declarator(parse_ctx *p);
int assignment_operator(parse_ctx *p);
int parameter_list(parse_ctx *p);
int parameter_declaration(parse_ctx *p);
int expression(parse_ctx *p);
int assignment_expression(parse_ctx *p);
int conditional_expression(parse_ctx *p);
int logical_or_expression(parse_ctx *p);
int logical_and_expression(parse_ctx *p);
int inclusive_or_expression(parse_ctx *p);
int exclusive_or_expression(parse_ctx *p);
int and_expression(parse_ctx *p);
int equality_expression(parse_ctx *p);
int relational_expression(parse_ctx *p);
int shift_expression(parse_ctx *p);
int additive_expression(parse_ctx *p);
int multiplicative_expression(parse_ctx *p);
int cast_expression(parse_ctx *p);
int unary_expression(parse_ctx *p);
int postfix_expression(parse_ctx *p);
int unary_operator(parse_ctx *p);
int primary_expression(parse_ctx *p);
int argument_expression_list(parse_ctx *p);
int constant(parse_ctx *p);
int string(parse_ctx *p);
int generic_selection(parse_ctx *p);
int generic_assoc_list(parse_ctx *p);
int generic_association(parse_ctx *p);
int designation(parse_ctx *p);
int designator_list(parse_ctx *p);
int designator(parse_ctx *p);
int statement(parse_ctx *p);
int labeled_statement(parse_ctx *p);
int labeled_statement(parse_ctx *p);
int expression_statement(parse_ctx *p);
int selection_statement(parse_ctx *p);
int iteration_statement(parse_ctx *p);
int jump_statement(parse_ctx *p);

// -----------------------------------------------------------------------

int declaration(parse_ctx *p) { assert(0); }
int declarator(parse_ctx *p) { assert(0); }
int declaration_list(parse_ctx *p) { assert(0); }
int compound_statement(parse_ctx *p) { assert(0); }
int declaration_or_statement(parse_ctx *p) { assert(0); }
int init_declarator_list(parse_ctx *p) { assert(0); }
int init_declarator(parse_ctx *p) { assert(0); }
int static_assert_declaration(parse_ctx *p) { assert(0); }
int typedef_name(parse_ctx *p) { assert(0); }
int function_specifier(parse_ctx *p) { assert(0); }
int alignment_specifier(parse_ctx *p) { assert(0); }
int pointer(parse_ctx *p) { assert(0); }
int direct_declarator(parse_ctx *p) { assert(0); }
int identifier_list(parse_ctx *p) { assert(0); }
int initializer_list(parse_ctx *p) { assert(0); }
int designative_initializer(parse_ctx *p) { assert(0); }
int initializer(parse_ctx *p) { assert(0); }
int constant_expression(parse_ctx *p) { assert(0); }
int struct_or_union_specifier(parse_ctx *p) { assert(0); }
int struct_or_union(parse_ctx *p) { assert(0); }
int struct_declaration_list(parse_ctx *p) { assert(0); }
int struct_declaration(parse_ctx *p) { assert(0); }
int enum_specifier(parse_ctx *p) { assert(0); }
int enumerator_list(parse_ctx *p) { assert(0); }
int enumerator(parse_ctx *p) { assert(0); }
int enumeration_constant(parse_ctx *p) { assert(0); }
int type_qualifier_list(parse_ctx *p) { assert(0); }
int parameter_type_list(parse_ctx *p) { assert(0); }
int struct_declarator(parse_ctx *p) { assert(0); }
int assignment_operator(parse_ctx *p) { assert(0); }
int parameter_list(parse_ctx *p) { assert(0); }
int parameter_declaration(parse_ctx *p) { assert(0); }
int expression(parse_ctx *p) { assert(0); }
int assignment_expression(parse_ctx *p) { assert(0); }
int conditional_expression(parse_ctx *p) { assert(0); }
int logical_or_expression(parse_ctx *p) { assert(0); }
int logical_and_expression(parse_ctx *p) { assert(0); }
int inclusive_or_expression(parse_ctx *p) { assert(0); }
int exclusive_or_expression(parse_ctx *p) { assert(0); }
int and_expression(parse_ctx *p) { assert(0); }
int equality_expression(parse_ctx *p) { assert(0); }
int relational_expression(parse_ctx *p) { assert(0); }
int shift_expression(parse_ctx *p) { assert(0); }
int additive_expression(parse_ctx *p) { assert(0); }
int multiplicative_expression(parse_ctx *p) { assert(0); }
int cast_expression(parse_ctx *p) { assert(0); }
int unary_expression(parse_ctx *p) { assert(0); }
int postfix_expression(parse_ctx *p) { assert(0); }
int unary_operator(parse_ctx *p) { assert(0); }
int primary_expression(parse_ctx *p) { assert(0); }
int argument_expression_list(parse_ctx *p) { assert(0); }
int constant(parse_ctx *p) { assert(0); }
int string(parse_ctx *p) { assert(0); }
int generic_selection(parse_ctx *p) { assert(0); }
int generic_assoc_list(parse_ctx *p) { assert(0); }
int generic_association(parse_ctx *p) { assert(0); }
int designation(parse_ctx *p) { assert(0); }
int designator_list(parse_ctx *p) { assert(0); }
int designator(parse_ctx *p) { assert(0); }
int statement(parse_ctx *p) { assert(0); }
int labeled_statement(parse_ctx *p) { assert(0); }
int expression_statement(parse_ctx *p) { assert(0); }
int selection_statement(parse_ctx *p) { assert(0); }
int iteration_statement(parse_ctx *p) { assert(0); }
int jump_statement(parse_ctx *p) { assert(0); }

// -----------------------------------------------------------------------

int storage_class_specifier(parse_ctx *p)
{
	int match = eat_identifier(p, "typedef")
	|| eat_identifier(p, "extern")
	|| eat_identifier(p, "static")
	|| eat_identifier(p, "_Thread_local")
	|| eat_identifier(p, "auto")
	|| eat_identifier(p, "register");

	if(match) {
		printf("storage_class_specifier: ");
		fwrite(p->tokens[-1].string, 1, p->tokens[-1].string_len, stdout);
		printf("\n");
	}

	return match;
}

int type_qualifier(parse_ctx *p)
{
	int match = eat_identifier(p, "const")
	|| eat_identifier(p, "restrict")
	|| eat_identifier(p, "volatile")
	|| eat_identifier(p, "_Atomic");

	if(match) {
		printf("type_qualifier: ");
		fwrite(p->tokens[-1].string, 1, p->tokens[-1].string_len, stdout);
		printf("\n");
	}

	return match;
}

int specifier_qualifier(parse_ctx *p)
{
	return type_specifier(p) || type_qualifier(p);
}

int specifier_qualifier_list(parse_ctx *p)
{
	int match = specifier_qualifier(p);
	if(match) while(specifier_qualifier(p)){}
	return match;
}

int direct_abstract_declarator(parse_ctx *p)
{
	parse_ctx p_saved = *p;

	if (eat_token(p, '(') 
		&& abstract_declarator(p)
		&& eat_token(p, ')')) {  return 1; }

	RESET();
	if (eat_token(p, '(') 
		&& parameter_type_list(p)
		&& eat_token(p, ')')) {  return 1; }

	RESET();
	if (eat_token(p, '(')
		&& eat_token(p, ')')) {  return 1; }

	RESET();
	if (eat_token(p, '[')
		&& OPTIONAL(eat_token(p, '*'))
		&& eat_token(p, ']')) {  return 1; }

	RESET();
	if (eat_token(p, '[')
		&& eat_identifier(p, "static")
		&& OPTIONAL(type_qualifier_list(p))
		&& assignment_expression(p) 
		&& eat_token(p, ']')) {  return 1; }

	RESET();
	if (eat_token(p, '[')
		&& type_qualifier_list(p) 
		&& OPTIONAL( 
			OPTIONAL(eat_identifier(p, "static")) 
			&& assignment_expression(p) )
		&& eat_token(p, ']')) {  return 1; }

	RESET();
	if (eat_token(p, '[')
		&& assignment_expression(p)
		&& eat_token(p, ']')) {  return 1; }

	RESET();
	if (direct_abstract_declarator(p)
		&& eat_token(p, '[')
		&& OPTIONAL(eat_token(p, '*'))
		&& eat_token(p, ']')) {  return 1; }

	RESET();
	if (direct_abstract_declarator(p)
		&& eat_token(p, '[')
		&& eat_identifier(p, "static")
		&& OPTIONAL(type_qualifier_list(p))
		&& assignment_expression(p)
		&& eat_token(p, ']')) {  return 1; }

	RESET();
	if (direct_abstract_declarator(p)
		&& eat_token(p, '[')
		&& type_qualifier_list(p)
		&& OPTIONAL( 
			OPTIONAL(eat_identifier(p, "static"))
			&& assignment_expression(p) )
		&& eat_token(p, ']')) {  return 1; }

	RESET();
	if (direct_abstract_declarator(p)
		&& eat_token(p, '[')
		&& assignment_expression(p)
		&& eat_token(p, ']')) {  return 1; }

	RESET();
	if (direct_abstract_declarator(p)
		&& eat_token(p, '(')
		&& parameter_type_list(p)
		&& eat_token(p, ')')) {  return 1; }

	RESET();
	if (direct_abstract_declarator(p)
		&& eat_token(p, '(')
		&& eat_token(p, ')')) {  return 1; }

	RESET();
	return 0;
}

int struct_declarator_list(parse_ctx *p)
{
	int match = struct_declarator(p);
	if(match) {
		parse_ctx copy = *p;
		while(eat_token(&copy,',') && struct_declarator(&copy))
			*p = copy;
	}
	return match;
}

int abstract_declarator(parse_ctx *p)
{
	if (pointer(p)) {
		(void) direct_abstract_declarator(p);
		return 1;
	}
	return direct_abstract_declarator(p);
}

int type_name(parse_ctx *p)
{
	int match = specifier_qualifier_list(p);
	if (match) { (void) abstract_declarator(p); }
	return match;
}

int atomic_type_specifier(parse_ctx *p)
{
	parse_ctx p_saved = *p;

	int match =  eat_identifier(p, "_Atomic")
	&& eat_token(p, '(')
	&& type_name(p) 
	&& eat_token(p, ')');

	if(!match) RESET();

	return match;
}

int type_specifier(parse_ctx *p)
{
	if(p->tokens == p->tokens_end) return 0;

	if (atomic_type_specifier(p)) return 1;
	else if (struct_or_union_specifier(p)) return 1;
	else if (enum_specifier(p)) return 1;
	else if (typedef_name(p)) return 1;

	else if (p->tokens[0].toktype == CLEX_id) {

		int match = eat_identifier(p, "void")
		|| eat_identifier(p, "char")
		|| eat_identifier(p, "short")
		|| eat_identifier(p, "int")
		|| eat_identifier(p, "long")
		|| eat_identifier(p, "float")
		|| eat_identifier(p, "double")
		|| eat_identifier(p, "signed")
		|| eat_identifier(p, "unsigned")
		|| eat_identifier(p, "_Bool")
		|| eat_identifier(p, "_Complex")
		|| eat_identifier(p, "_Imaginary");

		if(match) {
			printf("type_specifier: ");
			fwrite(p->tokens[-1].string, 1, p->tokens[-1].string_len, stdout);
			printf("\n");
			return 1;
		}
	}
	return 0;
}

int declaration_specifier(parse_ctx *p)
{
	return storage_class_specifier(p) 
	|| type_specifier(p)
	|| type_qualifier(p)
	|| function_specifier(p)
	|| alignment_specifier(p);
}

int declaration_specifiers(parse_ctx *p) 
{
	int match = declaration_specifier(p);
	if(match) while (declaration_specifier(p));
	return match;
}


int function_definition(parse_ctx *p) 
{
	parse_ctx p_saved = *p; 

	int match = declaration_specifiers(p);
	match = match && declarator(p);
	if(match) declaration_list(p);
	match = match && compound_statement(p); 
	
	if(!match) RESET();
	return match;
}

int external_declaration (parse_ctx *p) 
{
	return function_definition(p) || declaration(p);
}

void translation_unit(parse_ctx p)
{
	while(p.tokens < p.tokens_end) {
		external_declaration(&p);
	}
}

/*
	==========================================================
		Main
	==========================================================
*/


int 
main (const char *filetext, char *argv[])
{
	
	/*
		Figure out what file we're asked to work on
		if -f flag is supplied, obey that, 
		else use stdin
	*/
	
	char *filename = 0;
	char **arg = argv;
	while(*(++arg)) {
		if (!strcmp("-f",*arg)) {
			if(!++arg) die("got -f flag but no filename follows");
			filename = *arg;
		}
	}
	
	FILE *f = stdin;
	if(filename) {
		f = fopen(filename, "rb");
		if(!f) die("couldn't open '%s'", filename);
	}
	
	/*
		Wrapgen flags are:
		-e,arg,chkfn add custom error checking (call chkfn and pass arg (specified by number))
	*/
	
	
	/*
		Allocate buffers
	*/
	
	char  *text         = malloc(1<<27);
	char  *string_store = malloc(0x10000);
	token *tokens       = malloc((1<<27) * sizeof(*tokens));
	struct ht stringtable = {
		.ht = malloc((1<<28) * sizeof(char*)),
		.exp = 28,
	};
	int ntok = 0;
	
	if(!(text && string_store && tokens && stringtable.ht)) die("out of mem");
	
	/*
		Read input file
	*/
	
	long long len = fread(text, 1, 1<<27, f);
	if(len == (1<<27)) die("input file too long");
	fclose(f);
	
	/*
		Lex whole file
	*/
	
	stb_lexer lex = {0};
	stb_c_lexer_init(&lex, text, text+len, (char *) string_store, 0x10000);
	while(stb_c_lexer_get_token(&lex)) {
		token t = {.toktype = lex.token};
		switch(lex.token)
		{
			case CLEX_id: 
			case CLEX_dqstring:
			case CLEX_sqstring: 		
				t.string_len = strlen(lex.string);	
				t.string = intern(&stringtable, lex.string, t.string_len);
				break;
			case CLEX_charlit:
				t.string = lex.string;
				t.string_len = 1;
				break;
			case CLEX_intlit:
				t.int_number = lex.int_number;
				break;
			case CLEX_floatlit:
				t.real_number = lex.real_number;
				break;
			case CLEX_eq:
			case CLEX_noteq:
			case CLEX_lesseq:
			case CLEX_greatereq:
			case CLEX_andand:
			case CLEX_oror:
			case CLEX_shl:
			case CLEX_shr:
			case CLEX_plusplus:
			case CLEX_minusminus:
			case CLEX_arrow:
			case CLEX_andeq:
			case CLEX_oreq:
			case CLEX_xoreq:
			case CLEX_pluseq:
			case CLEX_minuseq:
			case CLEX_muleq:
			case CLEX_diveq:
			case CLEX_modeq:
			case CLEX_shleq:
			case CLEX_shreq:
			case CLEX_eqarrow:
				break;
			default:
				if (!(lex.token >= 0 && lex.token < 256)) {
					stb_lex_location loc = {0};
					stb_c_lexer_get_location(&lex, lex.where_firstchar, &loc);
					die("Lex error at line %i, character %i: unknown token %ld", loc.line_number, loc.line_offset, lex.token);
				}
				break;
		}		
		
		tokens[ntok++] = t;
	}
	
	/*
		Run parser on token array.
	*/

	long delimiterstack[4096];
	
	parse_ctx p = {
		.tokens      =  tokens,
		.tokens_end  =  tokens+ntok,
	};

	translation_unit(p);
}
