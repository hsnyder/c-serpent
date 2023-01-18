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
	int explicit_auto : 1;
	int is_typedef : 1;
	int is_extern : 1;
	int is_static : 1;
	int is_threadlocal : 1;
	int is_register : 1;
	int is_inline : 1; 
} DeclarationSpecifierInfo;

enum type_category {
	T_UNINITIALIZED = 0,
	T_UNKNOWN,
	T_CHAR,
	T_SHORT,
	T_INT,
	T_LONG,
	T_LLONG,
	T_FLOAT,
	T_DOUBLE,
	T_LDOUBLE,
	T_STRUCT,
	T_UNION,
	T_VOID,
	T_BOOL,
};

const char *type_category_strings[] = {
	[T_UNINITIALIZED] = "<uninitialized>",
	[T_UNKNOWN] = "<unknown>",
	[T_VOID] = "void",
	[T_BOOL] = "_Bool",
	[T_CHAR] = "char",
	[T_SHORT] = "short",
	[T_INT] = "int",
	[T_LONG] = "long",
	[T_LLONG] = "long long",
	[T_FLOAT] = "float",
	[T_DOUBLE] = "double",
	[T_LDOUBLE] = "long double",
	[T_STRUCT] = "struct",
	[T_UNION] = "union",
}

typedef struct type {
	enum type_category category;
	unsigned short is_function : 1;
	unsigned short is_unsigned : 1;
	unsigned short explicit_signed : 1;
	unsigned short is_imaginary : 1;
	unsigned short is_complex : 1;

	unsigned short is_array    : 1;
	unsigned short is_const    : 1;
	unsigned short is_restrict : 1;
	unsigned short is_volatile : 1;
	unsigned short is_pointer  : 1;

	/*
		If is_pointer is set, then look at pointer_levels[0].
		If pointer_levels[0].is_pointer is set, then look at [1], etc.

		We can only hold 3 levels of pointer in this type - sufficient for wrapgen.
	*/

	struct {
		unsigned char is_array    : 1;
		unsigned char is_const    : 1;
		unsigned char is_restrict : 1;
		unsigned char is_volatile : 1;
		unsigned char is_pointer  : 1;
       	} pointer_levels[3];
} Type;

typedef enum symbol_category {
	S_TYPEDEF,
	S_LABEL,
	S_DEFINITION,
	S_DECLARATION,
} SymbolCategory;

typedef struct {
	char *name;
	Type type;
	enum symbol_category category;
} Symbol;

typedef struct { // lexer token

	long toktype; // this will be one of the enum values in stb_c_lexer
	int string_len;
	union {
		double real_number;
		long long int_number;
		char * string;
	};
} Token;

typedef struct {
	Token     *tokens;   
	Token     *tokens_end;
	Token     *tokens_first;   
	int 	depth ;
} ParseCtx;

/*
	==========================================================
		Helper functions
	==========================================================
*/

#define COUNT_ARRAY(x) ((int64_t)(sizeof(x)/sizeof(x[0])))

#define OPTIONAL(x) ((x), 1)
#define RESTORE(p)  (*p = p_saved);
#define SAVE(p) ParseCtx p_saved = *p; 
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

void modify_type_struct(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category == T_UNKNOWN) return;
	if(s->type.category == T_UNINITIALIZED) s->type.category = T_STRUCT;
	else die(p, "'struct' does not make sense with '%s'", type_category_strings[s->type.category]);
}

void modify_type_union(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category == T_UNKNOWN) return;
	if(s->type.category == T_UNINITIALIZED) s->type.category = T_UNION;
	else die(p, "'union' does not make sense with '%s'", type_category_strings[s->type.category]);
}

void modify_type_void(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category == T_UNKNOWN) return;

	if(s->type.is_unsigned)     die(p, "'void' does not make sense with 'unsigned'");
	if(s->type.explicit_signed) die(p, "'void' does not make sense with 'signed'");
	if(s->type.is_imaginary)     die(p, "'void' does not make sense with '_Imaginary'");
	if(s->type.is_complex)       die(p, "'void' does not make sense with '_Complex'");

	if(s->type.category == T_UNINITIALIZED) s->type.category = T_VOID;
	else die(p, "'void' does not make sense with '%s'", type_category_strings[s->type.category]);
}

void modify_type_char(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category == T_UNKNOWN) return;
	if(s->type.is_imaginary)     die(p, "'char' does not make sense with '_Imaginary'");
	if(s->type.is_complex)       die(p, "'char' does not make sense with '_Complex'");
	if(s->type.category == T_UNINITIALIZED) s->type.category = T_CHAR;
	else die(p, "'char' does not make sense with '%s'", type_category_strings[s->type.category]);
}

void modify_type_short(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category == T_UNKNOWN) return;
	if(s->type.is_imaginary)     die(p, "'short' does not make sense with '_Imaginary'");
	if(s->type.is_complex)       die(p, "'short' does not make sense with '_Complex'");
	if(s->type.category == T_UNINITIALIZED || s->type.category == T_INT) s->type.category = T_SHORT;
	else die(p, "'short' does not make sense with '%s'", type_category_strings[s->type.category]);
}

void modify_type_int(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category == T_UNKNOWN) return;
	if(s->type.is_imaginary)     die(p, "'int' does not make sense with '_Imaginary'");
	if(s->type.is_complex)       die(p, "'int' does not make sense with '_Complex'");
	if(s->type.category == T_SHORT 
		|| s->type.category == T_LONG 
		|| s->type.category = T_LLONG ) return;

	if(s->type.category == T_UNINITIALIZED) s->type.category = T_INT;
	else die(p, "'int' does not make sense with '%s'", type_category_strings[s->type.category]);
}

void modify_type_long(ParseCtx *p, Symbol *s) 
{
	assert(s);
	if(s->type.category == T_UNKNOWN) return;
	if(s->type.category == T_UNINITIALIZED) s->type.category = T_VOID;
	else if (s->type.category == T_INT) s->type.category = T_LONG;
	else if (s->type.category == T_LONG) s->type.category = T_LLONG;
	else if (s->type.category == T_DOUBLE) s->type.category = T_LDOUBLE;
	else die(p, "'long' does not make sense with '%s'", type_category_strings[s->type.category]);
}

void modify_type_float(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category == T_UNKNOWN) return;
	if(s->type.is_unsigned)     die(p, "'float' does not make sense with 'unsigned'");
	if(s->type.explicit_signed) die(p, "'float' does not make sense with 'signed'");
	if(s->type.category == T_UNINITIALIZED) s->type.category = T_FLOAT;
	else die(p, "'float' does not make sense with '%s'", type_category_strings[s->type.category]);
}

void modify_type_double(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category == T_UNKNOWN) return;
	if(s->type.is_unsigned)     die(p, "'double' does not make sense with 'unsigned'");
	if(s->type.explicit_signed) die(p, "'double' does not make sense with 'signed'");
	if(s->type.category == T_UNINITIALIZED) s->type.category = T_DOUBLE;
	if(s->type.category == T_LONG) s->type.category = T_LDOUBLE;
	else die(p, "'double' does not make sense with '%s'", type_category_strings[s->type.category]);
}

void modify_type_signed(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category >= T_FLOAT) die(p, "'signed' doesn't make sense with non-integer types");
	if(s->type.is_unsigned) die(p, "'signed' doesn't make sense with 'unsigned'");
	s->type.explicit_signed = 1;
}

void modify_type_unsigned(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category >= T_FLOAT) die(p, "'unsigned' doesn't make sense with non-integer types");
	if(s->type.is_unsigned) die(p, "'unsigned' doesn't make sense with 'signed'");
	s->type.is_unsigned = 1;
}

void modify_type_bool(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category == T_UNKNOWN) return;
	if(s->type.is_unsigned)     die(p, "'_Bool' does not make sense with 'unsigned'");
	if(s->type.explicit_signed) die(p, "'_Bool' does not make sense with 'signed'");
	if(s->type.category == T_UNINITIALIZED) s->type.category = T_BOOL;
	else die(p, "'_Bool' does not make sense with '%s'", type_category_strings[s->type.category]);
}

void modify_type_complex(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category >= T_CHAR && T_s->type.category < T_FLOAT) die(p, "'_Complex' doesn't make sense with integer types");
	if(s->type.is_imaginary) die(p, "'_Complex' doesn't make sense with '_Imaginary'");
	s->type.is_complex = 1;
}

void modify_type_imaginary(ParseCtx *p, Symbol *s)
{
	assert(s);
	if(s->type.category >= T_CHAR && T_s->type.category < T_FLOAT) die(p, "'_Imaginary' doesn't make sense with integer types");
	if(s->type.is_complex) die(p, "'_Imaginary' doesn't make sense with '_Complex'");
	s->type.is_imaginary = 1;
}



static void repr_token(int bufsz, char buf[], Token t)
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

void dump_context(FILE *f, ParseCtx *p)
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
die (ParseCtx *p, const char * fmt, ...)
{
	va_list va;
	va_start(va, fmt);
	vfprintf(stderr, fmt, va);
	va_end(va);
	fprintf(stderr, "\n");
	if(p) dump_context(stderr, p);
	exit(EXIT_FAILURE);
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
	if(len == 0) len = strlen(str);
	char *x = malloc(len+1);
	if(!x) die(0, "strdup_len_or_die: out of memory");
	memcpy(x,str,len+1);
	return x;
}

char *intern(char *key, int keylen)
{
	// NOTE/TODO not thread safe (fine in this application) 
	
	static char *stringheap[1<<28] = {0};
	static struct ht stringtable = {
		.ht = stringheap,
		.exp = 28,
	};

	struct ht *t = &stringtable;

	if (keylen == 0) keylen = strlen(key);
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


// TODO/NOTE not thread safe
// NOTE this is only a global symbol table
Symbol symtab[100000] = {0};
int nsym = 0;

Symbol *add_symbol(Symbol s)
{
	if(nsym == COUNT_ARRAY(symtab)) die(0, "global symbol table full");
	s.name = intern(s.name, 0);
	symtab[nsym] = s;
	return &symtab[nsym++];
}

Symbol *get_symbol(char *name)
{
	for(int i = 0; i < nsym; i++)
		if(!strcmp(name,symtab[i].name)) return &symtab[i];
	return 0;
}


/*
	==========================================================
		C99 parsing
	==========================================================
*/


int eat_identifier(ParseCtx *p, const char *id)
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

int eat_token(ParseCtx *p, long toktype)
{
	if(p->tokens == p->tokens_end) return 0;
	if(p->tokens[0].toktype == toktype) {
		p->tokens++;
		return 1;
	}
	return 0;
}


// -----------------------------------------------------------------------


int external_declaration(ParseCtx *p);
int function_definition(ParseCtx *p);
int declaration(ParseCtx *p);
int declaration_specifiers(ParseCtx *p, DeclarationSpecifierInfo *dsi, Symbol *s);
int declaration_specifier(ParseCtx *p, DeclarationSpecifierInfo *dsi, Symbol *s);
int declarator(ParseCtx *p);
int declaration_list(ParseCtx *p);
int compound_statement(ParseCtx *p);
int declaration_or_statement(ParseCtx *p);
int init_declarator_list(ParseCtx *p);
int init_declarator(ParseCtx *p);
int static_assert_declaration(ParseCtx *p);
int storage_class_specifier(ParseCtx *p, DeclarationSpecifierInfo *dsi);
int type_specifier(ParseCtx *p, DeclarationSpecifierInfo *dsi, Symbol *s);
int typedef_name(ParseCtx *p, Symbol *out);
int type_qualifier(ParseCtx *p, Symbol *s);
int function_specifier(ParseCtx *p, DeclarationSpecifierInfo *dsi);
int alignment_specifier(ParseCtx *p);
int pointer(ParseCtx *p);
int direct_declarator(ParseCtx *p);
int identifier_list(ParseCtx *p);
int initializer_list(ParseCtx *p);
int designative_initializer(ParseCtx *p);
int initializer(ParseCtx *p);
int constant_expression(ParseCtx *p);
int atomic_type_specifier(ParseCtx *p);
int struct_or_union_specifier(ParseCtx *p, Symbol *s);
int struct_or_union(ParseCtx *p, Symbol *s);
int struct_declaration_list(ParseCtx *p);
int struct_declarator_list(ParseCtx *p);
int struct_declaration(ParseCtx *p);
int enum_specifier(ParseCtx *p);
int enumerator_list(ParseCtx *p);
int enumerator(ParseCtx *p);
int enumeration_constant(ParseCtx *p);
int type_name(ParseCtx *p);
int specifier_qualifier_list(ParseCtx *p, DeclarationSpecifierInfo *dsi);
int specifier_qualifier(ParseCtx *p, DeclarationSpecifierInfo *dsi);
int abstract_declarator(ParseCtx *p);
int direct_abstract_declarator(ParseCtx *p);
int type_qualifier_list(ParseCtx *p);
int parameter_type_list(ParseCtx *p);
int struct_declarator(ParseCtx *p);
int assignment_operator(ParseCtx *p);
int parameter_list(ParseCtx *p);
int parameter_declaration(ParseCtx *p);
int expression(ParseCtx *p);
int assignment_expression(ParseCtx *p);
int conditional_expression(ParseCtx *p);
int logical_or_expression(ParseCtx *p);
int logical_and_expression(ParseCtx *p);
int inclusive_or_expression(ParseCtx *p);
int exclusive_or_expression(ParseCtx *p);
int and_expression(ParseCtx *p);
int equality_expression(ParseCtx *p);
int relational_expression(ParseCtx *p);
int shift_expression(ParseCtx *p);
int additive_expression(ParseCtx *p);
int multiplicative_expression(ParseCtx *p);
int cast_expression(ParseCtx *p);
int unary_expression(ParseCtx *p);
int postfix_expression(ParseCtx *p);
int unary_operator(ParseCtx *p);
int primary_expression(ParseCtx *p);
int argument_expression_list(ParseCtx *p);
int constant(ParseCtx *p);
int string(ParseCtx *p);
int generic_selection(ParseCtx *p);
int generic_assoc_list(ParseCtx *p);
int generic_association(ParseCtx *p);
int designation(ParseCtx *p);
int designator_list(ParseCtx *p);
int designator(ParseCtx *p);
int statement(ParseCtx *p);
int labeled_statement(ParseCtx *p);
int labeled_statement(ParseCtx *p);
int expression_statement(ParseCtx *p);
int selection_statement(ParseCtx *p);
int iteration_statement(ParseCtx *p);
int jump_statement(ParseCtx *p);

int string_literal(ParseCtx *p);
int integer_constant(ParseCtx *p);
int character_constant(ParseCtx *p);
int floating_constant(ParseCtx *p);
int identifier(ParseCtx *p);



// ------------------------------------------------------------------



void translation_unit(ParseCtx p)
{
	while(p.tokens < p.tokens_end) {
		if(!external_declaration(&p)) 
			die(&p, "Failed to parse before end of file");
	}
}

int external_declaration (ParseCtx *p) 
{
	return function_definition(p) || declaration(p);
}

int function_definition(ParseCtx *p) 
{
	SAVE(p);

	DeclarationSpecifierInfo dsi = {0};
	Symbol s = {0};

	int match = declaration_specifiers(p, &dsi, &s);
	match = match && declarator(p);
	if(match) declaration_list(p);
	match = match && compound_statement(p); 
	
	if(!match) RESTORE(p);
	return match;
}

int declaration(ParseCtx *p)
{
	SAVE(p);
	DeclarationSpecifierInfo dsi = {0};
	Symbol s = {0};

	if (declaration_specifiers(p, &dsi, &s)
		&& OPTIONAL(init_declarator_list(p))
		&& eat_token(p, ';')) { return 1; }

	RESTORE(p);

	if (static_assert_declaration(p)) { return 1; }

	if (eat_token(p, ';')) { return 1; }

	return 0;
}

int declaration_specifiers(ParseCtx *p, DeclarationSpecifierInfo *dsi, Symbol *s) 
{
	int match = declaration_specifier(p, dsi, s);
	if(match) while (declaration_specifier(p, dsi, s));
	return match;
}

int declaration_specifier(ParseCtx *p, DeclarationSpecifierInfo *dsi, Symbol *s)
{
	return storage_class_specifier(p, dsi) 
	|| type_specifier(p, dsi, s)
	|| type_qualifier(p, s)
	|| function_specifier(p, dsi)
	|| alignment_specifier(p);
}

int declarator(ParseCtx *p) 
{
	(void) pointer(p);
	return direct_declarator(p);
}

int declaration_list(ParseCtx *p)
{
	int match = declaration(p);
	if(match) while(declaration(p)){}
	return match;
}

int compound_statement(ParseCtx *p)
{
	SAVE(p);

	int match = eat_token(p, '{');
	if(match) {
		p->depth++;
		while(declaration_or_statement(p)) {}
		p->depth--;
	}
	match = match && eat_token(p, '}');

	if(!match) RESTORE(p);
	return match;
}

int declaration_or_statement(ParseCtx *p)
{
	return declaration(p) || statement(p);
}

int init_declarator_list(ParseCtx *p)
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

int init_declarator(ParseCtx *p)
{
	int match = declarator(p);
	if (match) {
		SAVE(p);
		if(eat_token(p,'=') && initializer(p)) {}
		else RESTORE(p);
	}

	return match;
	
}

int static_assert_declaration(ParseCtx *p)
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

int storage_class_specifier(ParseCtx *p, DeclarationSpecifierInfo *dsi)
{
	if(eat_identifier(p, "typedef")) { if(dsi)dsi->is_typedef = 1; return 1; }
	if(eat_identifier(p, "extern"))  { if(dsi)dsi->is_extern = 1; return 1; }
	if(eat_identifier(p, "static"))  { if(dsi)dsi->is_static = 1; return 1; }
	if(eat_identifier(p, "_Thread_local"))  { if(dsi)dsi->is_threadlocal = 1; return 1; }
	if(eat_identifier(p, "auto"))  { if(dsi)dsi->explicit_auto = 1; return 1; }
	if(eat_identifier(p, "register"))  { if(dsi)dsi->is_register = 1; return 1; }
	return 0;
}

int type_specifier(ParseCtx *p, DeclarationSpecifierInfo *dsi, Symbol *s)
{
	if(p->tokens == p->tokens_end) return 0;

	if (atomic_type_specifier(p)) return 1;
	else if (struct_or_union_specifier(p, s)) { return 1; }
	else if (enum_specifier(p)) { modify_type_int(p,s); return 1; }
	else if (typedef_name(p, s)) return 1;

	else if (p->tokens[0].toktype == CLEX_id) {

		if (eat_identifier(p, "void"))     { modify_type_void(p,s); return 1; }
		if (eat_identifier(p, "char"))     { modify_type_char(p,s); return 1; } 
		if (eat_identifier(p, "short"))    { modify_type_short(p,s); return 1; }
		if (eat_identifier(p, "int"))      { modify_type_int(p,s); return 1; }
		if (eat_identifier(p, "long"))     { modify_type_long(p,s); return 1; }
		if (eat_identifier(p, "float"))    { modify_type_float(p,s); return 1; }
		if (eat_identifier(p, "double"))   { modify_type_double(p,s); return 1; }
		if (eat_identifier(p, "signed"))   { modify_type_signed(p,s); return 1; }
		if (eat_identifier(p, "unsigned")) { modify_type_unsigned(p,s); return 1; }
		if (eat_identifier(p, "_Bool"))      { modify_type_bool(p,s); return 1; }
		if (eat_identifier(p, "_Complex"))    { modify_type_complex(p,s); return 1; }
		if (eat_identifier(p, "_Imaginary"))  { modify_type_imaginary(p,s); return 1; }

		return 0;
	}
	return 0;
}

int typedef_name(ParseCtx *p, Symbol *out)
{
	if(p->tokens == p->tokens_end) return 0;
	if(p->tokens[0].toktype == CLEX_id) {

		Symbol *s = 0;
		if ((s = get_symbol(p->tokens[0].string))) {
			if (s->category == S_TYPEDEF) {
				*out = *s;
				p->tokens++;
				return 1;
			}
		}
	}
	return 0;
}

int type_qualifier(ParseCtx *p)
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

int function_specifier(ParseCtx *p, DeclarationSpecifierInfo *dsi)
{
	if(eat_identifier(p, "inline")) { if(dsi)dsi->is_inline = 1; return 1; }
	if(eat_identifier(p, "_Noreturn")) return 1;
	return 0;
}

int alignment_specifier(ParseCtx *p)
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

int pointer(ParseCtx *p)
{
	int match = eat_token(p, '*');
	if (match) {
		(void) type_qualifier_list(p);
		(void) pointer(p);
	}
	return match;
}

int direct_declarator_head(ParseCtx *p) 
{
	SAVE(p);

	if (identifier(p)) { return 1; }

	if (eat_token(p, '(')
		&& declarator(p)
		&& eat_token(p, ')')) { return 1; }

	RESTORE(p);
	return 0;
}

int direct_declarator_tail(ParseCtx *p) 
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

int direct_declarator(ParseCtx *p)
{
	if(!direct_declarator_head(p)) return 0;
	while(direct_declarator_tail(p)) { }

	return 1;
}

int identifier_list(ParseCtx *p)
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

int initializer_list(ParseCtx *p)
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

int designative_initializer(ParseCtx *p)
{
	(void) designation(p);
	return initializer(p);
}

int initializer (ParseCtx *p)
{
	SAVE(p);

	if (eat_token(p, '{')
		&& initializer_list(p)
		&& OPTIONAL(eat_token(p, ','))
		&& eat_token(p, '}')) { return 1; }

	RESTORE(p);
	
	return assignment_expression(p);
}

int constant_expression(ParseCtx *p)
{
	// TODO add constraints!
	return conditional_expression(p);
}

int atomic_type_specifier(ParseCtx *p)
{
	SAVE(p);

	int match =  eat_identifier(p, "_Atomic")
	&& eat_token(p, '(')
	&& type_name(p) 
	&& eat_token(p, ')');

	if(!match) RESTORE(p);

	return match;
}

int struct_or_union_specifier(ParseCtx *p, Symbol *s)
{
	SAVE(p);

	if (struct_or_union(p, s)
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

int struct_or_union(ParseCtx *p, Symbol *s)
{
	if(eat_identifier(p, "struct")) { modify_type_struct(p, s); return 1; }
	if(eat_identifier(p, "union")) { modify_type_union(p, s); return 1; }
	return 0;
}

int struct_declaration_list(ParseCtx *p)
{
	int match = struct_declaration(p);
	if(match) while(struct_declaration(p)) { }
	return match;
}

int struct_declaration(ParseCtx *p)
{
	SAVE(p);
	DeclarationSpecifierInfo dsi = {0};

	// anonymous struct/union
	if (specifier_qualifier_list(p, &dsi) 
		&& eat_token(p, ';')) {return 1;}

	RESTORE(p);
	if (specifier_qualifier_list(p, &dsi)
		&& struct_declarator_list(p)
		&& eat_token(p, ';')) {return 1;}

	RESTORE(p);
	if (static_assert_declaration(p)) {return 1;}

	RESTORE(p);
	return 0;
}

int enum_specifier(ParseCtx *p)
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

int enumerator_list(ParseCtx *p)
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

int enumerator(ParseCtx *p)
{
	int match = enumeration_constant(p);
	if(match) {
		SAVE(p);
		if (eat_token(p, '=') && constant_expression(p)) { }
		else RESTORE(p);
	}
	return match;
}

int enumeration_constant(ParseCtx *p)
{
	// todo add some checking?
	return identifier(p);
}

int type_name(ParseCtx *p)
{
	DeclarationSpecifierInfo dsi = {0};
	int match = specifier_qualifier_list(p, &dsi);
	if (match) { (void) abstract_declarator(p); }
	return match;
}

int specifier_qualifier_list(ParseCtx *p, DeclarationSpecifierInfo *dsi)
{
	int match = specifier_qualifier(p, dsi);
	if(match) while(specifier_qualifier(p, dsi)){}
	return match;
}

int specifier_qualifier(ParseCtx *p, DeclarationSpecifierInfo *dsi)
{
	return type_specifier(p, dsi) || type_qualifier(p);
}


int abstract_declarator(ParseCtx *p)
{
	if (pointer(p)) {
		(void) direct_abstract_declarator(p);
		return 1;
	}
	return direct_abstract_declarator(p);
}

int direct_abstract_declarator_head(ParseCtx *p)
{
	SAVE(p);

	if (eat_token(p, '(') 
		&& abstract_declarator(p)
		&& eat_token(p, ')')) {  return 1; }

	RESTORE(p);
	return 0; 
}

int direct_abstract_declarator_tail(ParseCtx *p)
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

int direct_abstract_declarator(ParseCtx *p)
{
	int did_consume_some_input = direct_abstract_declarator_head(p);
	while(direct_abstract_declarator_tail(p)) { did_consume_some_input = 1;}	

	return did_consume_some_input;
}

int struct_declarator_list(ParseCtx *p)
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

int type_qualifier_list(ParseCtx *p)
{
	int match = type_qualifier(p);
	if(match) while(type_qualifier(p)) {}
	return match;
}

int parameter_type_list(ParseCtx *p)
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

int struct_declarator(ParseCtx *p)
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

int assignment_operator(ParseCtx *p)
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


int parameter_list(ParseCtx *p)
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

int parameter_declaration(ParseCtx *p)
{
	DeclarationSpecifierInfo dsi = {0};
	int match = declaration_specifiers(p, &dsi);
	if(match) {
		if(declarator(p) || abstract_declarator(p)) { }
	}
	return match;
}

int expression(ParseCtx *p)
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

int assignment_expression(ParseCtx *p) 
{
	if(conditional_expression(p)) return 1;

	SAVE(p);

	if (unary_expression(p) 
		&& assignment_operator(p)
		&& assignment_expression(p)) { return 1; }

	RESTORE(p);
	return 0;
}

int conditional_expression(ParseCtx *p)
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

int logical_or_expression(ParseCtx *p)
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

int logical_and_expression(ParseCtx *p)
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

int inclusive_or_expression(ParseCtx *p)
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

int exclusive_or_expression(ParseCtx *p)
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

int and_expression(ParseCtx *p)
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

int equality_expression(ParseCtx *p)
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


int relational_expression(ParseCtx *p)
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

int shift_expression(ParseCtx *p)
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


int additive_expression(ParseCtx *p)
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


int multiplicative_expression(ParseCtx *p)
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

int cast_expression(ParseCtx *p)
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

int unary_expression(ParseCtx *p)
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

int postfix_expression_head(ParseCtx *p)
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


int postfix_expression_tail(ParseCtx *p)
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

int postfix_expression(ParseCtx *p)
{
	if(!postfix_expression_head(p)){return 0;}
	while(postfix_expression_tail(p)) { }
	return 1;
}

int unary_operator(ParseCtx *p)
{
	return eat_token(p, '&')
	|| eat_token(p, '*')
	|| eat_token(p, '+')
	|| eat_token(p, '-')
	|| eat_token(p, '~')
	|| eat_token(p, '!');
}

int primary_expression(ParseCtx *p)
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

int argument_expression_list(ParseCtx *p)
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

int constant(ParseCtx *p)
{
	return integer_constant(p)
	|| character_constant(p)
	|| floating_constant(p)
	|| enumeration_constant(p);
}

int string(ParseCtx *p)
{
	return string_literal(p) 
	|| eat_identifier(p, "__func__");
}


int generic_selection(ParseCtx *p)
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

int generic_assoc_list(ParseCtx *p)
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

int generic_association(ParseCtx *p)
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

int designation(ParseCtx *p)
{
	SAVE(p);
	int match = designator_list(p)
		&& eat_token(p, '=');
	if(!match) RESTORE(p);
	return match;
}

int designator_list(ParseCtx *p)
{
	int match = designator(p);
	if(match) while(designator(p)) {}
	return match;
}

int designator(ParseCtx *p)
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

int statement(ParseCtx *p)
{
	return labeled_statement(p)
	|| compound_statement(p)
	|| expression_statement(p)
	|| selection_statement(p)
	|| iteration_statement(p)
	|| jump_statement(p);
}

int labeled_statement(ParseCtx *p)
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

int expression_statement(ParseCtx *p)
{
	(void) expression(p);
	return eat_token(p, ';');
}

int selection_statement(ParseCtx *p)
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

int iteration_statement(ParseCtx *p)
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

int jump_statement(ParseCtx *p)
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



int string_literal(ParseCtx *p)
{
	return eat_token(p, CLEX_dqstring);
}

int integer_constant(ParseCtx *p)
{
	return eat_token(p, CLEX_intlit);
}

int character_constant(ParseCtx *p)
{
	return eat_token(p, CLEX_charlit);
}

int floating_constant(ParseCtx *p)
{
	return eat_token(p, CLEX_floatlit);
}

int identifier(ParseCtx *p)
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
	Token *tokens       = malloc((1<<27) * sizeof(*tokens));

	int ntok = 0;
	
	if(!(text && string_store && tokens)) die(0,"out of mem");
	
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
		Token t = {.toktype = lex.token};
		switch(lex.token)
		{
			case CLEX_id: 
			case CLEX_dqstring:
			case CLEX_sqstring: 		
				t.string_len = strlen(lex.string);	
				t.string = intern(lex.string, t.string_len);
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

	ParseCtx p = {
		.tokens_first  =   tokens,
		.tokens        =   tokens,
		.tokens_end    =   tokens+ntok,
	};

	translation_unit(p);
}
