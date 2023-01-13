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

typedef struct {
	long toktype; // this will be one of the enum values in stb_c_lexer
	int string_len;
	union {
		double real_number;
		long long int_number;
		char * string;
	};
} token;


typedef struct {
	int sym_len;
	char *symbol;
} typedef_entry;

typedef struct {
	typedef_entry *td;	
} typedef_table;

typedef struct {
	token     *tokens;   
	token     *tokens_end;
	token     *tokens_first;   
	typedef_table     ttab;
} parse_ctx;

/*
	==========================================================
		Helper functions
	==========================================================
*/

#define COUNT_ARRAY(x) ((int64_t)(sizeof(x)/sizeof(x[0])))

#define OPTIONAL(x) ((x), 1)
#define RESTORE(p)  (*p = p_saved);
#define SAVE(p) parse_ctx p_saved = *p; 
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

static void repr_token(int bufsz, char buf[], token t)
{
	switch (t.toktype) {
		case CLEX_id        : snprintf(buf, bufsz,"%s", t.string); break;
		case CLEX_eq        : snprintf(buf, bufsz,"=="); break;
		case CLEX_noteq     : snprintf(buf, bufsz,"!="); break;
		case CLEX_lesseq    : snprintf(buf, bufsz,"<="); break;
		case CLEX_greatereq : snprintf(buf, bufsz,">="); break;
		case CLEX_andand    : snprintf(buf, bufsz,"&&"); break;
		case CLEX_oror      : snprintf(buf, bufsz,"||"); break;
		case CLEX_shl       : snprintf(buf, bufsz,"<<"); break;
		case CLEX_shr       : snprintf(buf, bufsz,">>"); break;
		case CLEX_plusplus  : snprintf(buf, bufsz,"++"); break;
		case CLEX_minusminus: snprintf(buf, bufsz,"--"); break;
		case CLEX_arrow     : snprintf(buf, bufsz,"->"); break;
		case CLEX_andeq     : snprintf(buf, bufsz,"&="); break;
		case CLEX_oreq      : snprintf(buf, bufsz,"|="); break;
		case CLEX_xoreq     : snprintf(buf, bufsz,"^="); break;
		case CLEX_pluseq    : snprintf(buf, bufsz,"+="); break;
		case CLEX_minuseq   : snprintf(buf, bufsz,"-="); break;
		case CLEX_muleq     : snprintf(buf, bufsz,"*="); break;
		case CLEX_diveq     : snprintf(buf, bufsz,"/="); break;
		case CLEX_modeq     : snprintf(buf, bufsz,"%%="); break;
		case CLEX_shleq     : snprintf(buf, bufsz,"<<="); break;
		case CLEX_shreq     : snprintf(buf, bufsz,">>="); break;
		case CLEX_eqarrow   : snprintf(buf, bufsz,"=>"); break;
		case CLEX_dqstring  : snprintf(buf, bufsz,"\"%s\"", t.string); break;
		case CLEX_sqstring  : snprintf(buf, bufsz,"'\"%s\"'", t.string); break;
		case CLEX_charlit   : snprintf(buf, bufsz,"'%s'", t.string); break;
		case CLEX_intlit    : snprintf(buf, bufsz,"#%lli", t.int_number); break;
		case CLEX_floatlit  : snprintf(buf, bufsz,"%g", t.real_number); break;
		default:
				      if (t.toktype >= 0 && t.toktype < 256)
					      snprintf(buf, bufsz,"%c", (int) t.toktype);
				      else {
					      snprintf(buf, bufsz,"<<<UNKNOWN TOKEN %ld >>>\n", t.toktype);
				      }
				      break;
	}
}

void dump_context(FILE *f, parse_ctx *p)
{
	long long before = MIN(p->tokens - p->tokens_first, 20); 
	long long after  = MIN(p->tokens_end - p->tokens, 20);

	for (int i = -before; i < after; i++)
	{
		char buf[1000] = {0};
		repr_token(sizeof(buf), buf, p->tokens[i]);
		if( i == 0 )
			fprintf(f, "HERE>>> %s ", buf);
		else 
			fprintf(f, "%s ", buf);
	}
	fprintf(f,"\n");
}


_Noreturn void
die (parse_ctx *p, const char * fmt, ...)
{
	va_list va;
	va_start(va, fmt);
	vfprintf(stderr, fmt, va);
	va_end(va);
	fprintf(stderr, "\n");
	if(p) dump_context(stderr, p);
	exit(EXIT_FAILURE);
}

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
	
	if (!n) die(0, "couldn't parse '%6s' as an integer", save);

	if (nchars_read) *nchars_read = x-save;
	return v * sign;

overflow:
	die(0, "integer overflow when trying to convert '%14s'", save);
}


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
	if(!x) die(0, "strdup_len_or_die: out of memory");
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
				die(0, "out of memory in intern");
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


#include "buf.h"

int check_typedef(parse_ctx *p, char *symbol, int sym_len)
{
	assert(symbol);
	if (sym_len < 1) sym_len = strlen(symbol);

	for(int i = 0; i < buf_size(p->ttab.td); i++) 
		if (sym_len == p->ttab.td[i].sym_len && 
			memcmp(p->ttab.td[i].symbol, symbol, sym_len)) { return 1; }
	return 0;
}

void add_typedef(parse_ctx *p, char *symbol, int sym_len)
{
	assert(symbol);
	if (sym_len < 1) sym_len = strlen(symbol);

	if(!check_typedef(p, symbol, sym_len))
		buf_push(p->ttab.td, 
			((typedef_entry){
				.sym_len = sym_len,
				.symbol  = strdup_len_or_die(symbol, sym_len),
			}));
}


/*
	==========================================================
		C99 parsing
	==========================================================
*/


int eat_identifier(parse_ctx *p, const char *id)
{
	if(p->tokens == p->tokens_end) return 0;
	if(p->tokens[0].toktype != CLEX_id) return 0;
	if((long long)strlen(id) != p->tokens[0].string_len) return 0;
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
int struct_declarator_list(parse_ctx *p);
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

int string_literal(parse_ctx *p);
int integer_constant(parse_ctx *p);
int character_constant(parse_ctx *p);
int floating_constant(parse_ctx *p);
int identifier(parse_ctx *p);



// ------------------------------------------------------------------



void translation_unit(parse_ctx p)
{
	while(p.tokens < p.tokens_end) {
		if(!external_declaration(&p)) 
			die(&p, "Failed to parse before end of file");
	}
}

int external_declaration (parse_ctx *p) 
{
	return function_definition(p) || declaration(p);
}

int function_definition(parse_ctx *p) 
{
	SAVE(p);

	int match = declaration_specifiers(p);
	match = match && declarator(p);
	if(match) declaration_list(p);
	match = match && compound_statement(p); 
	
	if(!match) RESTORE(p);
	return match;
}

int declaration(parse_ctx *p)
{
	SAVE(p);
	if (declaration_specifiers(p)
		&& OPTIONAL(init_declarator_list(p))
		&& eat_token(p, ';')) { return 1; }
	RESTORE(p);

	if (static_assert_declaration(p)) { return 1; }

	if (eat_token(p, ';')) { return 1; }

	return 0;
}

int declaration_specifiers(parse_ctx *p) 
{
	int match = declaration_specifier(p);
	if(match) while (declaration_specifier(p));
	return match;
}

int declaration_specifier(parse_ctx *p)
{
	return storage_class_specifier(p) 
	|| type_specifier(p)
	|| type_qualifier(p)
	|| function_specifier(p)
	|| alignment_specifier(p);
}

int declarator(parse_ctx *p) 
{
	(void) pointer(p);
	return direct_declarator(p);
}

int declaration_list(parse_ctx *p)
{
	int match = declaration(p);
	if(match) while(declaration(p)){}
	return match;
}

int compound_statement(parse_ctx *p)
{
	SAVE(p);

	int match = eat_token(p, '{');
	if(match) while(declaration_or_statement(p)) {}
	match = match && eat_token(p, '}');

	if(!match) RESTORE(p);
	return match;
}

int declaration_or_statement(parse_ctx *p)
{
	return declaration(p) || statement(p);
}

int init_declarator_list(parse_ctx *p)
{
	int match = init_declarator(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, ',') && init_declarator(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int init_declarator(parse_ctx *p)
{
	int match = declarator(p);
	if (match) {
		SAVE(p);
		if(eat_token(p,'=') && initializer(p)) {}
		else RESTORE(p);
	}

	return match;
	
}

int static_assert_declaration(parse_ctx *p)
{
	SAVE(p);

	int match = eat_identifier(p, "_Static_assert")
	&& eat_token(p, '(')
	&& constant_expression(p)
	&& eat_token(p, ',')
	&& string_literal(p)
	&& eat_token(p,')')
	&& eat_token(p,';');

	if(!match) RESTORE(p);
	return match;
}

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

int typedef_name(parse_ctx *p)
{
	if(p->tokens == p->tokens_end) return 0;
	if(p->tokens[0].toktype == CLEX_id) {

		if (check_typedef(p, p->tokens[0].string, p->tokens[0].string_len)) {
			p->tokens++;
			return 1;
		}
	}
	return 0;
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

int function_specifier(parse_ctx *p)
{
	int match = eat_identifier(p, "inline")
	|| eat_identifier(p, "_Noreturn");

	if(match){
		printf("function_specifier: ");
		fwrite(p->tokens[-1].string, 1, p->tokens[-1].string_len, stdout);
		printf("\n");
	}

	return match;
}

int alignment_specifier(parse_ctx *p)
{
	SAVE(p);
	
	if (eat_identifier(p, "_Alignas")
		&& eat_token(p, '(')
		&& type_name(p)
		&& eat_token(p, ')')) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "_Alignas")
		&& eat_token(p, '(')
		&& constant_expression(p)
		&& eat_token(p, ')')) { return 1; }

	RESTORE(p);
	return 0;
}

int pointer(parse_ctx *p)
{
	int match = eat_token(p, '*');
	if (match) {
		(void) type_qualifier_list(p);
		(void) pointer(p);
	}
	return match;
}

int direct_declarator_head(parse_ctx *p) 
{
	SAVE(p);

	if (identifier(p)) { return 1; }

	if (eat_token(p, '(')
		&& declarator(p)
		&& eat_token(p, ')')) { return 1; }

	RESTORE(p);
	return 0;
}

int direct_declarator_tail(parse_ctx *p) 
{
	SAVE(p);

	if (eat_token(p, '[')
		&& OPTIONAL(eat_token(p, '*'))
		&& eat_token(p, ']')) { return 1; }

	RESTORE(p);
	if (eat_token(p, '[')
		&& eat_identifier(p, "static")
		&& OPTIONAL(type_qualifier_list(p))
		&& assignment_expression(p)
		&& eat_token(p, ']')) { return 1; }

	RESTORE(p);
	if (eat_token(p, '[')
		&& type_qualifier_list(p)
		&& OPTIONAL(eat_token(p, '*'))
		&& eat_token(p, ']')) { return 1; }

	RESTORE(p);
	if (eat_token(p, '[')
		&& type_qualifier_list(p)
		&& OPTIONAL(eat_identifier(p, "static"))
		&& assignment_expression(p)
		&& eat_token(p, ']')) { return 1; }

	RESTORE(p);
	if (eat_token(p, '[')
		&& assignment_expression(p)
		&& eat_token(p, ']')) { return 1; }

	RESTORE(p);
	if (eat_token(p, '(')
		&& parameter_type_list(p)
		&& eat_token(p, ')')) { return 1; }

	RESTORE(p);
	if (eat_token(p, '(')
		&& identifier_list(p)
		&& eat_token(p, ')')) { return 1; }

	RESTORE(p);
	if (eat_token(p, '(')
		&& eat_token(p, ')')) { return 1; }

	RESTORE(p);
	return 0;

}

int direct_declarator(parse_ctx *p)
{
	if(!direct_declarator_head(p)) return 0;
	while(direct_declarator_tail(p)) { }

	return 1;
}

int identifier_list(parse_ctx *p)
{
	int match = identifier(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, ',') && identifier(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int initializer_list(parse_ctx *p)
{
	int match = designative_initializer(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, ',') && designative_initializer(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int designative_initializer(parse_ctx *p)
{
	(void) designation(p);
	return initializer(p);
}

int initializer (parse_ctx *p)
{
	SAVE(p);

	if (eat_token(p, '{')
		&& initializer_list(p)
		&& OPTIONAL(eat_token(p, ','))
		&& eat_token(p, '}')) { return 1; }

	RESTORE(p);
	
	return assignment_expression(p);
}

int constant_expression(parse_ctx *p)
{
	// TODO add constraints!
	return conditional_expression(p);
}

int atomic_type_specifier(parse_ctx *p)
{
	SAVE(p);

	int match =  eat_identifier(p, "_Atomic")
	&& eat_token(p, '(')
	&& type_name(p) 
	&& eat_token(p, ')');

	if(!match) RESTORE(p);

	return match;
}

int struct_or_union_specifier(parse_ctx *p)
{
	SAVE(p);

	if (struct_or_union(p)
		&& eat_token(p, '{')
		&& struct_declaration_list(p)
		&& eat_token(p, '}')) { return 1; }

	RESTORE(p);
	if (struct_or_union(p) && identifier(p)) 
	{ 

		SAVE(p);

		if (eat_token(p, '{')
			&& struct_declaration_list(p)
			&& eat_token(p, '}')) { }
		else RESTORE(p);

		return 1;
	}

	RESTORE(p);
	return 0;
}

int struct_or_union(parse_ctx *p)
{
	return eat_identifier(p, "struct") || eat_identifier(p, "union");
}

int struct_declaration_list(parse_ctx *p)
{
	int match = struct_declaration(p);
	if(match) while(struct_declaration(p)) { }
	return match;
}

int struct_declaration(parse_ctx *p)
{
	SAVE(p);

	// anonymous struct/union
	if (specifier_qualifier_list(p) 
		&& eat_token(p, ';')) {return 1;}

	RESTORE(p);
	if (specifier_qualifier_list(p)
		&& struct_declarator_list(p)
		&& eat_token(p, ';')) {return 1;}

	RESTORE(p);
	if (static_assert_declaration(p)) {return 1;}

	RESTORE(p);
	return 0;
}

int enum_specifier(parse_ctx *p)
{
	SAVE(p);

	if (eat_identifier(p, "enum") 
		&& eat_token(p, '{')
		&& enumerator_list(p)
		&& OPTIONAL(eat_token(p,','))
		&& eat_token(p, '}')) { return 1;}

	RESTORE(p);
	if (eat_identifier(p, "enum") 
		&& identifier(p)) 
	{

		SAVE(p);

		if(eat_token(p, '{')
			&& enumerator_list(p)
			&& OPTIONAL(eat_token(p,','))
			&& eat_token(p, '}')) { }
		else RESTORE(p);

		return 1;
	}
	return 0;
}

int enumerator_list(parse_ctx *p)
{
	int match = enumerator(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, ',') && enumerator(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int enumerator(parse_ctx *p)
{
	int match = enumeration_constant(p);
	if(match) {
		SAVE(p);
		if (eat_token(p, '=') && constant_expression(p)) { }
		else RESTORE(p);
	}
	return match;
}

int enumeration_constant(parse_ctx *p)
{
	// todo add some checking?
	return identifier(p);
}

int type_name(parse_ctx *p)
{
	int match = specifier_qualifier_list(p);
	if (match) { (void) abstract_declarator(p); }
	return match;
}

int specifier_qualifier_list(parse_ctx *p)
{
	int match = specifier_qualifier(p);
	if(match) while(specifier_qualifier(p)){}
	return match;
}

int specifier_qualifier(parse_ctx *p)
{
	return type_specifier(p) || type_qualifier(p);
}


int abstract_declarator(parse_ctx *p)
{
	if (pointer(p)) {
		(void) direct_abstract_declarator(p);
		return 1;
	}
	return direct_abstract_declarator(p);
}

int direct_abstract_declarator_head(parse_ctx *p)
{
	SAVE(p);

	if (eat_token(p, '(') 
		&& abstract_declarator(p)
		&& eat_token(p, ')')) {  return 1; }

	RESTORE(p);
	return 0; 
}

int direct_abstract_declarator_tail(parse_ctx *p)
{
	SAVE(p);
	if (eat_token(p, '[')
		&& OPTIONAL(type_qualifier_list(p))
		&& OPTIONAL(assignment_expression(p))
		&& eat_token(p, ']')) {  return 1; }

	RESTORE(p);
	if (eat_token(p, '[')
		&& eat_identifier(p, "static")
		&& OPTIONAL(type_qualifier_list(p))
		&& assignment_expression(p)
		&& eat_token(p, ']')) {  return 1; }

	RESTORE(p);
	if (eat_token(p, '[')
		&& type_qualifier_list(p)
		&& eat_identifier(p, "static")
		&& assignment_expression(p)
		&& eat_token(p, ']')) {  return 1; }

	RESTORE(p);
	if (eat_token(p, '[')
		&& eat_token(p, '*')
		&& eat_token(p, ']')) {  return 1; }

	RESTORE(p);
	if (eat_token(p, '(')
		&& OPTIONAL(parameter_type_list(p))
		&& eat_token(p, ')')) {  return 1; }

	RESTORE(p);
	return 0;

}

int direct_abstract_declarator(parse_ctx *p)
{
	int did_consume_some_input = direct_abstract_declarator_head(p);
	while(direct_abstract_declarator_tail(p)) { did_consume_some_input = 1;}	

	return did_consume_some_input;
}

int struct_declarator_list(parse_ctx *p)
{
	int match = struct_declarator(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, ',') && struct_declarator(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int type_qualifier_list(parse_ctx *p)
{
	int match = type_qualifier(p);
	if(match) while(type_qualifier(p)) {}
	return match;
}

int parameter_type_list(parse_ctx *p)
{
	int match = parameter_list(p);
	if(match){
		SAVE(p);
		// BUG: three dots don't need to be adjacent here
		if (eat_token(p,',') 
			&& eat_token(p, '.')
			&& eat_token(p, '.')
			&& eat_token(p, '.')) { }
		else RESTORE(p);
	}
	return match;
}

int struct_declarator(parse_ctx *p)
{
	SAVE(p);

	if (eat_token(p, ':') && constant_expression(p)) { return 1; }

	RESTORE(p);
	if (declarator(p)) {
		SAVE(p);

		if (eat_token(p, ':') && constant_expression(p)) { }
		else RESTORE(p);

		return 1;
	}

	RESTORE(p);
	return 0;
}

int assignment_operator(parse_ctx *p)
{
	return eat_token(p, '=')
	|| eat_token(p, CLEX_muleq)
	|| eat_token(p, CLEX_diveq)
	|| eat_token(p, CLEX_modeq)
	|| eat_token(p, CLEX_pluseq)
	|| eat_token(p, CLEX_minuseq)
	|| eat_token(p, CLEX_shleq)
	|| eat_token(p, CLEX_shreq)
	|| eat_token(p, CLEX_andeq)
	|| eat_token(p, CLEX_xoreq)
	|| eat_token(p, CLEX_oreq);
}


int parameter_list(parse_ctx *p)
{
	int match = parameter_declaration(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, ',') && parameter_declaration(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int parameter_declaration(parse_ctx *p)
{
	int match = declaration_specifiers(p);
	if(match) {
		if(declarator(p) || abstract_declarator(p)) { }
	}
	return match;
}

int expression(parse_ctx *p)
{
	int match = assignment_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, ',') && assignment_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int assignment_expression(parse_ctx *p) 
{
	if(conditional_expression(p)) return 1;

	SAVE(p);

	if (unary_expression(p) 
		&& assignment_operator(p)
		&& assignment_expression(p)) { return 1; }

	RESTORE(p);
	return 0;
}

int conditional_expression(parse_ctx *p)
{
	int match = logical_or_expression(p);
	if (match) {
		SAVE(p);
		if (eat_token(p,'?')
			&& expression(p)
			&& eat_token(p, ':')
			&& conditional_expression(p)) { }
		else RESTORE(p);
	}
	return match;
}

int logical_or_expression(parse_ctx *p)
{
	int match = logical_and_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, CLEX_oror) && logical_and_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int logical_and_expression(parse_ctx *p)
{
	int match = inclusive_or_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, CLEX_andand) && inclusive_or_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int inclusive_or_expression(parse_ctx *p)
{
	int match = exclusive_or_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, '|') && exclusive_or_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int exclusive_or_expression(parse_ctx *p)
{
	int match = and_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, '^') && and_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int and_expression(parse_ctx *p)
{
	int match = equality_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, '&') && equality_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int equality_expression(parse_ctx *p)
{
	int match = relational_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if ((eat_token(p, CLEX_eq) || eat_token(p, CLEX_noteq)) 
			&& relational_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}


int relational_expression(parse_ctx *p)
{
	int match = shift_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if ((eat_token(p, '<') || eat_token(p, '>') || eat_token(p, CLEX_lesseq) || eat_token(p, CLEX_greatereq)) 
			&& shift_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int shift_expression(parse_ctx *p)
{
	int match = additive_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if ((eat_token(p, CLEX_shl) || eat_token(p, CLEX_shr)) 
			&& additive_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}


int additive_expression(parse_ctx *p)
{
	int match = multiplicative_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if ((eat_token(p, '+') || eat_token(p, '-')) 
			&& multiplicative_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}


int multiplicative_expression(parse_ctx *p)
{
	int match = cast_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if ((eat_token(p, '*') || eat_token(p, '/') || eat_token(p, '%') )
			&& cast_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int cast_expression(parse_ctx *p)
{
	if (unary_expression(p)) { return 1; }

	SAVE(p);

	if (eat_token(p, '(')
		&& type_name(p)
		&& eat_token(p, ')')
		&& cast_expression(p)) { return 1; }
	
	RESTORE(p);
	return 0;
}

int unary_expression(parse_ctx *p)
{
	if (postfix_expression(p)) { return 1; }

	SAVE(p);

	if ((eat_token(p, CLEX_plusplus) || eat_token(p, CLEX_minusminus))
		&& unary_expression(p)) { return 1;}

	RESTORE(p);
	if (unary_operator(p) 
		&& cast_expression(p)) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "sizeof")
		&& unary_expression(p)) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "sizeof")
		&& eat_token(p, '(')
		&& type_name(p)
		&& eat_token(p, ')')) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "_Alignof")
		&& eat_token(p, '(')
		&& type_name(p)
		&& eat_token(p, ')')) { return 1; }

	RESTORE(p);
	return 0;
}

int postfix_expression_head(parse_ctx *p)
{
	if (primary_expression(p)) { return 1; }
	
	SAVE(p);
	if (eat_token(p, '(')
		&& type_name(p)
		&& eat_token(p, ')')
		&& eat_token(p, '{')
		&& initializer_list(p)
		&& OPTIONAL(eat_token(p, ','))
		&& eat_token(p, '}')) { return 1; }

	RESTORE(p);
	return 0;
}


int postfix_expression_tail(parse_ctx *p)
{
	SAVE(p);
	if (eat_token(p, '[')
		&& expression(p)
		&& eat_token(p, ']')) { return 1; }

	RESTORE(p);
	if (eat_token(p, '(')
		&& OPTIONAL(argument_expression_list(p))
		&& eat_token(p, ')')) { return 1; }

	RESTORE(p);
	if ((eat_token(p,'.') || eat_token(p, CLEX_arrow))
		&& identifier(p)) { return 1; }

	RESTORE(p);
	if (eat_token(p, CLEX_plusplus) || eat_token(p, CLEX_minusminus)) { return 1; }

	RESTORE(p);
	return 0;
}

int postfix_expression(parse_ctx *p)
{
	if(!postfix_expression_head(p)){return 0;}
	while(postfix_expression_tail(p)) { }
	return 1;
}

int unary_operator(parse_ctx *p)
{
	return eat_token(p, '&')
	|| eat_token(p, '*')
	|| eat_token(p, '+')
	|| eat_token(p, '-')
	|| eat_token(p, '~')
	|| eat_token(p, '!');
}

int primary_expression(parse_ctx *p)
{
	if (identifier(p)) { return 1; } 

	if (constant(p)) { return 1; }

	if (string(p)) { return 1; }

	SAVE(p);
	if (eat_token(p, '(')
		&& expression(p)
		&& eat_token(p, ')')) { return 1;}
	RESTORE(p);

	if (generic_selection(p)) { return 1; }

	return 0;
}

int argument_expression_list(parse_ctx *p)
{
	int match = assignment_expression(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, ',') && assignment_expression(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int constant(parse_ctx *p)
{
	return integer_constant(p)
	|| character_constant(p)
	|| floating_constant(p)
	|| enumeration_constant(p);
}

int string(parse_ctx *p)
{
	return string_literal(p) 
	|| eat_identifier(p, "__func__");
}


int generic_selection(parse_ctx *p)
{
	SAVE(p);
	int match = eat_identifier(p, "_Generic")
		&& eat_token(p, '(')
		&& assignment_expression(p)
		&& eat_token(p, ',')
		&& generic_assoc_list(p)
		&& eat_token(p, ')');

	if(!match) RESTORE(p);
	return match;
}

int generic_assoc_list(parse_ctx *p)
{
	int match = generic_association(p);

	if(match) while(1) 
	{
		SAVE(p);
		if (eat_token(p, ',') && generic_association(p)) {}
		else {
			RESTORE(p);
			break;
		}
	}

	return match;
}

int generic_association(parse_ctx *p)
{
	SAVE(p);

	if (type_name(p)
		&& eat_token(p, ':')
		&& assignment_expression(p)) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "default")
		&& eat_token(p, ':')
		&& assignment_expression(p)) { return 1; }

	RESTORE(p);
	return 0;
}

int designation(parse_ctx *p)
{
	SAVE(p);
	int match = designator_list(p)
		&& eat_token(p, '=');
	if(!match) RESTORE(p);
	return match;
}

int designator_list(parse_ctx *p)
{
	int match = designator(p);
	if(match) while(designator(p)) {}
	return match;
}

int designator(parse_ctx *p)
{
	SAVE(p);

	if (eat_token(p, '[')
		&& constant_expression(p)
		&& eat_token(p,']')) { return 1; }

	RESTORE(p);
	if (eat_token(p, '.')
		&& identifier(p)) { return 1; }

	RESTORE(p);
	return 0;
}

int statement(parse_ctx *p)
{
	return labeled_statement(p)
	|| compound_statement(p)
	|| expression_statement(p)
	|| selection_statement(p)
	|| iteration_statement(p)
	|| jump_statement(p);
}

int labeled_statement(parse_ctx *p)
{
	SAVE(p);

	if (identifier(p)
		&& eat_token(p, ':')
		&& statement(p)) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "case")
		&& constant_expression(p)
		&& eat_token(p, ':')
		&& statement(p)) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "default")
		&& eat_token(p, ':')
		&& statement(p)) { return 1; }

	RESTORE(p);
	return 0;
}

int expression_statement(parse_ctx *p)
{
	(void) expression(p);
	return eat_token(p, ';');
}

int selection_statement(parse_ctx *p)
{
	SAVE(p);
	
	if (eat_identifier(p, "if") 
		&& eat_token(p, '(')
		&& expression(p)
		&& eat_token(p, ')')
		&& statement(p)
		&& eat_identifier(p, "else")
		&& statement(p)) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "if") 
		&& eat_token(p, '(')
		&& expression(p)
		&& eat_token(p, ')')
		&& statement(p)) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "switch")
		&& eat_token(p, '(')
		&& expression(p)
		&& eat_token(p, ')')
		&& statement(p)) { return 1; }

	RESTORE(p);
	return 0;
}

int iteration_statement(parse_ctx *p)
{
	SAVE(p);

	if (eat_identifier(p, "while")
		&& eat_token(p, '(')
		&& expression(p)
		&& eat_token(p, ')')
		&& statement(p)) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "do")
		&& statement(p)
		&& eat_identifier(p, "while")
		&& eat_token(p, '(')
		&& expression(p)
		&& eat_token(p, ')')
		&& eat_token(p, ';')) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "for")
		&& eat_token(p, '(')
		&& OPTIONAL(expression(p))
		&& eat_token(p, ';')
		&& OPTIONAL(expression(p))
		&& eat_token(p, ';')
		&& OPTIONAL(expression(p))
		&& eat_token(p, ')')
		&& statement(p)) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "for")
		&& eat_token(p, '(')
		&& declaration(p)
		&& OPTIONAL(expression(p))
		&& eat_token(p, ';')
		&& OPTIONAL(expression(p))
		&& eat_token(p, ')')
		&& statement(p)) { return 1; }

	RESTORE(p);
	return 0;
}

int jump_statement(parse_ctx *p)
{
	SAVE(p);

	if (eat_identifier(p, "goto")
		&& identifier(p)
		&& eat_token(p, ';')) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "continue")
		&& eat_token(p, ';')) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "break")
		&& eat_token(p, ';')) { return 1; }

	RESTORE(p);
	if (eat_identifier(p, "return")
		&& OPTIONAL(expression(p))
		&& eat_token(p, ';')) { return 1; }

	RESTORE(p);
	return 0;
}



int string_literal(parse_ctx *p)
{
	return eat_token(p, CLEX_dqstring);
}

int integer_constant(parse_ctx *p)
{
	return eat_token(p, CLEX_intlit);
}

int character_constant(parse_ctx *p)
{
	return eat_token(p, CLEX_charlit);
}

int floating_constant(parse_ctx *p)
{
	return eat_token(p, CLEX_floatlit);
}

int identifier(parse_ctx *p)
{
	return eat_token(p, CLEX_id);
}


/*
	==========================================================
		Main
	==========================================================
*/

#ifdef _WIN32
#define popen(x,y) _popen(x,y)
#define pclose(x) _pclose(x)
#endif

int 
main (int argc, char *argv[])
{
	(void)argc;
	
	/*
		Figure out what file we're asked to work on
		if -f flag is supplied, obey that, 
		else use stdin
	*/

	char *filename = 0;
	char **arg = argv;
	while(*(++arg)) {
		if (!strcmp("-f",*arg)) {
			if(!++arg) die(0, "got -f flag but no filename follows");
			filename = *arg;
		}
	}

	if(filename) die(0,"-f flag not yet implemented");

	FILE *f = stdin;
	if(filename) {

		f = fopen(filename, "rb");
		if(!f) die(0,"couldn't open '%s'", filename);

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
	
	if(!(text && string_store && tokens && stringtable.ht)) die(0,"out of mem");
	
	/*
		Read input file
	*/
	
	long long len = fread(text, 1, 1<<27, f);
	if(len == (1<<27)) die(0,"input file too long");
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
					die(0,"Lex error at line %i, character %i: unknown token %ld", loc.line_number, loc.line_offset, lex.token);
				}
				break;
		}		
		
		tokens[ntok++] = t;
	}
	
	/*
		Run parser on token array.
	*/

	parse_ctx p = {
		.tokens_first  =   tokens,
		.tokens        =   tokens,
		.tokens_end    =   tokens+ntok,
	};

	translation_unit(p);
}
