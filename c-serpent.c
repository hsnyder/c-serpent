/*
	Harris M. Snyder, 2023
	This is free and unencumbered software released into the public domain.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <limits.h>


#define STB_C_LEXER_IMPLEMENTATION
#include "stb_c_lexer.h"

#define MAX_FN_ARGS 40
#define MAX_DIRS 40
#define MAX_FILES 40

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
};

typedef struct type {
	enum type_category category;
	short explicit_signed : 1;
	short is_unsigned  : 1;
	short is_complex   : 1;
	short is_imaginary : 1;
	short is_const     : 1;
	short is_restrict  : 1;
	short is_volatile  : 1;
	short is_pointer          : 1;
	short is_pointer_const    : 1;
	short is_pointer_restrict : 1;
	short is_pointer_volatile : 1;
} Type;

typedef struct {
	char *name;
	Type type;
} Symbol;

typedef struct {
	Type type;
	char suffix; 
	char found;
} VariantSuffix;

typedef struct { // lexer token

	long toktype; // this will be one of the enum values in stb_c_lexer
	int string_len;
	union {
		double real_number;
		long long int_number;
		char * string;
	};
} Token;

typedef struct
{
	int verbose;
	int disable_declarations;
	const char * modulename;
	const char * filename;
	const char * preprocessor;
	int disable_pp;
	int generic;
	int generic_keep_trailing_underscore;
	const char *dirs[MAX_DIRS];
	int ndirs;

	struct {
		const char *fn;
		short active;
	 	short argno;
	} error_handling;

	struct {
		const char *files[MAX_FILES];
		int nfiles;
	} manual_include;

} CSerpentArgs;
enum { 
	MAX_STRINGS_EXP=15, 
	MAX_STRING_HEAP=(1<<22),
};

typedef struct {
	// String table
	int num_strings, heap_size;
	char heap[MAX_STRING_HEAP];
	char *table[1<<MAX_STRINGS_EXP];

	// Symbol table
	int nsym;
	Symbol symbols[10000];
} StorageBuffers;

typedef struct {
	Token     *tokens;   
	Token     *tokens_end;
	Token     *tokens_first;   
	StorageBuffers *storage;
	CSerpentArgs args;
} ParseCtx;


/*
	==========================================================
		Helper functions for parsing and handling errors
	==========================================================
*/

#define ssizeof(x) ((int64_t)sizeof(x))
#define COUNT_ARRAY(x) ((int64_t)(sizeof(x)/sizeof(x[0])))

#define OPTIONAL(x) ((x), 1)
#define RESTORE(p)  (*p = p_saved);
#define SAVE(p) ParseCtx p_saved = *p; 
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

#ifdef CSERPENT_DISABLE_ASSERT
#  define assert(c)
#else
#  if defined(_MSC_VER)
#    define assert(c) if(!(c)){__debugbreak();}
#  else
#    if defined(__GNUC__) || defined(__clang__)
#      define assert(c) if(!(c)){__builtin_trap();}
#    else 
#      define assert(c) if(!(c)){*(volatile int*)0=0;}
#    endif 
#  endif
#endif

#define internal static
#define global static

internal int 
repr_type(int bufsz, char buf[], Type type) 
{

	char *is_signed    = type.explicit_signed ? "signed " : "";
        char *is_unsigned  = type.is_unsigned     ? "unsigned " : "";
	char *is_complex   = type.is_complex      ? "_Complex " : "";
	char *is_imaginary = type.is_imaginary    ? "_Imaginary " : "";
	char *is_const     = type.is_const        ? "const " : "";
	char *is_restrict  = type.is_restrict     ? "restrict " : "";
	char *is_volatile  = type.is_volatile     ? "volatile " : "";
	char *is_pointer   = type.is_pointer      ? "*" : "";

	char *is_pointer_const    = type.is_pointer_const    ? "const " : "";
	char *is_pointer_restrict = type.is_pointer_restrict ? "restrict " : "";
	char *is_pointer_volatile = type.is_pointer_volatile ? "volatile " : "";

	return snprintf(buf, bufsz, "%s%s%s %s%s%s%s%s%s%s%s%s", 
		is_signed, is_unsigned, type_category_strings[type.category], is_complex, is_imaginary,
		is_const, is_restrict, is_volatile, is_pointer,
		is_pointer_const, is_pointer_restrict, is_pointer_volatile);
}


internal int 
repr_symbol(int bufsz, char buf[], Symbol s) 
{

	long x = snprintf(buf, bufsz, "%s :=  ", s.name);
	bufsz -= x;
	buf += x;
	// TODO bug: make sure we haven't already overflown the buffer.

	return x + repr_type(bufsz, buf, s.type);
}

internal void 
repr_token(int bufsz, char buf[], Token t)
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

internal void 
dump_context(FILE *f, ParseCtx *p)
{
	long long before = MIN(p->tokens - p->tokens_first, 20); 
	long long after  = MIN(p->tokens_end - p->tokens, 20);

	for (int i = -before; i < after; i++)
	{
		char buf[1000] = {0};
		repr_token(sizeof(buf), buf, p->tokens[i]);
		if( i == 0 )
			fprintf(f, ">>HERE<< %s ", buf);
		else 
			fprintf(f, "%s ", buf);
	}
	fprintf(f,"\n");
}


internal _Noreturn void
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

internal uint64_t 
hash (char *s, int32_t len)
{
	uint64_t h = 0x100;
	for (int32_t i = 0; i < len; i++) {
		h ^= s[i] & 255;
		h *= 1111111111111111111;
	}
	return h ^ h>>32;
}

internal int32_t 
ht_lookup(uint64_t hash, int exp, int32_t idx)
{
	uint32_t mask = ((uint32_t)1 << exp) - 1;
	uint32_t step = (hash >> (64 - exp)) | 1;
	return (idx + step) & mask;
}

internal char *
intern_string(StorageBuffers *st, char *key, int keylen)
{
	if (keylen == 0) keylen = strlen(key);
	uint64_t h = hash(key, keylen+1);
	for (int32_t i = h;;) {
		i = ht_lookup(h, MAX_STRINGS_EXP, i);
		if (!st->table[i]) {
			// empty, insert here
			if (st->num_strings+1 == COUNT_ARRAY(st->table)/2)
				die(0, "intern: string table full");
			if (st->heap_size + keylen + 1 >= MAX_STRING_HEAP)
				die(0, "intern: string heap full");
			st->num_strings++;
			st->table[i] = st->heap+st->heap_size;
			memcpy(st->table[i], key, keylen);
			st->heap_size += keylen;
			st->heap[st->heap_size++] = 0;
			return st->table[i];
		} else if (!strcmp(st->table[i], key)) {
			// found, return canonical instance
			return st->table[i];
		}
	}
}

internal int 
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


internal Symbol *
add_symbol(StorageBuffers *storage, Symbol s)
{
	if(storage->nsym == COUNT_ARRAY(storage->symbols)) die(0, "symbol table full");
	s.name = intern_string(storage, s.name, 0);
	storage->symbols[storage->nsym] = s;
	return &storage->symbols[storage->nsym++];
}

internal Symbol *
get_symbol(StorageBuffers *storage, char *name)
{
	for(int i = 0; i < storage->nsym; i++)
		if(!strcmp(name,storage->symbols[i].name)) return &storage->symbols[i];
	return 0;
}

internal Symbol *
get_symbol_or_die(StorageBuffers *storage, char *name)
{
	Symbol *s = get_symbol(storage, name);
	if(!s) die(0, "Unknown type: %s", name);
	return s;
}

internal void 
clear_symbols(StorageBuffers *storage)
{
	storage->nsym = 0;
}

internal int 
modify_type_pointer(ParseCtx *p, Type *type)
{
	assert(type);
	if (type->is_pointer) 
		return 0;
	type->is_pointer = 1;
	return 1;
}

internal void 
modify_type_const(ParseCtx *p, Type *type)
{
	(void) p;
	assert(type);
	if(type->is_pointer) type->is_pointer_const = 1;
	else type->is_const = 1;
}

internal void 
modify_type_restrict(ParseCtx *p, Type *type)
{
	(void) p;
	assert(type);
	if(type->is_pointer) type->is_pointer_restrict = 1;
	else type->is_restrict = 1;
}

internal void 
modify_type_volatile(ParseCtx *p, Type *type)
{
	(void) p;
	assert(type);
	if(type->is_pointer) type->is_pointer_volatile = 1;
	else type->is_volatile = 1;
}

internal void 
modify_type_struct(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->category == T_UNINITIALIZED) type->category = T_STRUCT;
	else die(p, "'struct' does not make sense with '%s'", type_category_strings[type->category]);
}

internal void 
modify_type_union(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->category == T_UNINITIALIZED) type->category = T_UNION;
	else die(p, "'union' does not make sense with '%s'", type_category_strings[type->category]);
}

internal void 
modify_type_void(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;

	if(type->is_unsigned)     die(p, "'void' does not make sense with 'unsigned'");
	if(type->explicit_signed) die(p, "'void' does not make sense with 'signed'");
	if(type->is_imaginary)     die(p, "'void' does not make sense with '_Imaginary'");
	if(type->is_complex)       die(p, "'void' does not make sense with '_Complex'");

	if(type->category == T_UNINITIALIZED) type->category = T_VOID;
	else die(p, "'void' does not make sense with '%s'", type_category_strings[type->category]);
}

internal void 
modify_type_char(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->is_imaginary)     die(p, "'char' does not make sense with '_Imaginary'");
	if(type->is_complex)       die(p, "'char' does not make sense with '_Complex'");
	if(type->category == T_UNINITIALIZED) type->category = T_CHAR;
	else die(p, "'char' does not make sense with '%s'", type_category_strings[type->category]);
}

internal void 
modify_type_short(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->is_imaginary)     die(p, "'short' does not make sense with '_Imaginary'");
	if(type->is_complex)       die(p, "'short' does not make sense with '_Complex'");
	if(type->category == T_UNINITIALIZED || type->category == T_INT) type->category = T_SHORT;
	else die(p, "'short' does not make sense with '%s'", type_category_strings[type->category]);
}

internal void 
modify_type_int(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->is_imaginary)     die(p, "'int' does not make sense with '_Imaginary'");
	if(type->is_complex)       die(p, "'int' does not make sense with '_Complex'");
	if(type->category == T_SHORT 
		|| type->category == T_LONG 
		|| type->category == T_LLONG ) return;

	if(type->category == T_UNINITIALIZED) type->category = T_INT;
	else die(p, "'int' does not make sense with '%s'", type_category_strings[type->category]);
}

internal void 
modify_type_long(ParseCtx *p, Type *type) 
{
	assert(type);
	if(type->category == T_UNKNOWN) return;

	if(type->category == T_UNINITIALIZED) type->category = T_LONG;
	else if (type->category == T_INT) type->category = T_LONG;
	else if (type->category == T_LONG) type->category = T_LLONG;
	else if (type->category == T_DOUBLE) type->category = T_LDOUBLE;
	else die(p, "'long' does not make sense with '%s'", type_category_strings[type->category]);
}

internal void 
modify_type_float(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->is_unsigned)     die(p, "'float' does not make sense with 'unsigned'");
	if(type->explicit_signed) die(p, "'float' does not make sense with 'signed'");

	if(type->category == T_UNINITIALIZED) type->category = T_FLOAT;
	else die(p, "'float' does not make sense with '%s'", type_category_strings[type->category]);
}

internal void 
modify_type_double(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->is_unsigned)     die(p, "'double' does not make sense with 'unsigned'");
	if(type->explicit_signed) die(p, "'double' does not make sense with 'signed'");

	if(type->category == T_UNINITIALIZED) type->category = T_DOUBLE;
	else if(type->category == T_LONG) type->category = T_LDOUBLE;
	else die(p, "'double' does not make sense with '%s'", type_category_strings[type->category]);
}

internal void 
modify_type_signed(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category >= T_FLOAT) die(p, "'signed' doesn't make sense with non-integer types");
	if(type->is_unsigned) die(p, "'signed' doesn't make sense with 'unsigned'");
	type->explicit_signed = 1;
}

internal void 
modify_type_unsigned(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category >= T_FLOAT) die(p, "'unsigned' doesn't make sense with non-integer types");
	if(type->is_unsigned) die(p, "'unsigned' doesn't make sense with 'signed'");
	type->is_unsigned = 1;
}

internal void 
modify_type_bool(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->is_unsigned)     die(p, "'_Bool' does not make sense with 'unsigned'");
	if(type->explicit_signed) die(p, "'_Bool' does not make sense with 'signed'");

	if(type->category == T_UNINITIALIZED) type->category = T_BOOL;
	else die(p, "'_Bool' does not make sense with '%s'", type_category_strings[type->category]);
}

internal void 
modify_type_complex(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category >= T_CHAR && type->category < T_FLOAT) die(p, "'_Complex' doesn't make sense with integer types");
	if(type->is_imaginary) die(p, "'_Complex' doesn't make sense with '_Imaginary'");
	type->is_complex = 1;
	// TODO check against struct/union?
}

internal void 
modify_type_imaginary(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category >= T_CHAR && type->category < T_FLOAT) die(p, "'_Imaginary' doesn't make sense with integer types");
	if(type->is_complex) die(p, "'_Imaginary' doesn't make sense with '_Complex'");
	type->is_imaginary = 1;
	// TODO check against struct/union?
}

internal int 
compare_types_equal(Type a, Type b, int compare_pointer, int compare_const, int compare_volatile, int compare_restrict) 
{
	if(a.category != b.category) return 0;

	if(a.category == T_FLOAT || a.category == T_DOUBLE || a.category == T_LDOUBLE) {

		if  (a.is_complex   != b.is_complex)     return 0;
		if  (a.is_imaginary != b.is_imaginary)  return 0;

	} else if (a.category >= T_CHAR && a.category <= T_LLONG) {

		if (a.is_unsigned != b.is_unsigned) return 0;

	}

	if (compare_const) if (a.is_const != b.is_const) return 0;
	if (compare_volatile) if (a.is_volatile != b.is_volatile) return 0;
	if (compare_restrict) if (a.is_restrict != b.is_restrict) return 0;

	if (compare_pointer) {
		if (a.is_pointer != b.is_pointer) return 0;

		if (compare_const) if (a.is_pointer_const != b.is_pointer_const) return 0;
		if (compare_volatile) if (a.is_pointer_volatile != b.is_pointer_volatile) return 0;
		if (compare_restrict) if (a.is_pointer_restrict != b.is_pointer_restrict) return 0;
	}

	return 1;
}




/*
	==========================================================
		Templates
	==========================================================
*/



/*
	==========================================================
		Wrapper generation
	==========================================================
*/

internal void 
emit_module(CSerpentArgs flags, int n_fnames, const char *fnames[], _Bool fname_needs_strip_underscore[])
{
	if(!flags.modulename) return;

	printf("static PyMethodDef module_functions[] = { \n");

	for (int i = 0; i < n_fnames; i++) {
		char name[500] = {0};
		int len = snprintf(name, sizeof(name), "%s", fnames[i]);
		assert(ssizeof(name)-1 > len);
		if(fname_needs_strip_underscore[i]) name[len-1] = 0;

		printf("{\"%s\", (PyCFunction) wrap_%s, METH_VARARGS|METH_KEYWORDS, \"\"},\n", 
			name, fnames[i]);
	}

	printf(
	"	{ NULL, NULL, 0, NULL } \n"
	"}; \n"
	"\n\n"
	"static const char module_name[] = \"%s\"; \n"
	"\n"
	"static struct PyModuleDef module_def = { \n"
	"	PyModuleDef_HEAD_INIT, \n"
	"	module_name,      /* m_name */ \n"
	"	NULL,             /* m_doc */ \n"
	"	-1,               /* m_size */ \n"
	"	module_functions, /* m_methods */ \n"
	"	NULL,             /* m_reload */ \n"
	"	NULL,             /* m_traverse */ \n"
	"	NULL,             /* m_clear  */ \n"
	"	NULL,             /* m_free */ \n"
	"}; \n"
	" \n"
	"static PyObject *module_init(void) \n"
	"{ \n"
	"	PyObject *m; \n"
	" \n"
	"	// Import numpy arrays \n"
	"	import_array1(NULL); \n"
	" \n"
	"	// Register the module \n"
	"	if (!(m = PyModule_Create(&module_def))) \n"
	"		return NULL; \n"
	" \n"
	"	return m; \n"
	"} \n"
	" \n"
	"PyMODINIT_FUNC PyInit_%s (void) \n"
	"{ \n"
	"	return module_init(); \n"
	"} \n", flags.modulename, flags.modulename);
}

internal void 
emit_preamble(CSerpentArgs flags)
{
	(void) flags;
	global const char * preamble = 
	"#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION \n"
	"#define PY_ARRAY_UNIQUE_SYMBOL SHARED_ARRAY_ARRAY_API \n"
	"#include <Python.h> \n"
	"#include <numpy/arrayobject.h> \n"
	"#define C2NPY(type) _Generic((type){0},    \\\n"
	"	signed char:        NPY_BYTE,      \\\n"
	"	short:              NPY_SHORT,     \\\n"
	"	int:                NPY_INT,       \\\n"
	"	long:               NPY_LONG,      \\\n"
	"	long long:          NPY_LONGLONG,  \\\n"
	"	unsigned char:      NPY_UBYTE,     \\\n"
	"	unsigned short:     NPY_USHORT,    \\\n"
	"	unsigned int:       NPY_UINT,      \\\n"
	"	unsigned long:      NPY_ULONG,     \\\n"
	"	unsigned long long: NPY_ULONGLONG, \\\n"
	"	float:              NPY_FLOAT,     \\\n"
	"	double:             NPY_DOUBLE,    \\\n"
	"	_Complex float:     NPY_CFLOAT,    \\\n"
	"	_Complex double:    NPY_CDOUBLE    \\\n"
	"	)\n";
	printf("%s\n", preamble);
}

internal int 
is_string(Type t)
{
	return t.category == T_CHAR
		&& !t.explicit_signed
		&& !t.is_unsigned
		&& t.is_pointer;
}

internal int 
is_voidptr(Type t)
{
	return t.category == T_VOID
		&& t.is_pointer;
}

internal int 
is_plainvoid(Type t)
{
	Type zero = {0};
	if(t.category == T_VOID){
		t.category = 0;
		return !memcmp(&t,&zero,sizeof(t));
	}
	return 0;
}

internal int 
is_array(Type t)
{
	return !is_string(t) 
		&& !is_voidptr(t)
		&& t.is_pointer;
}

internal Type 
basetype(Type t)
{
	t.is_pointer = 0;
	t.is_const = 0;
	t.is_volatile = 0;
	t.is_restrict = 0;
	t.is_pointer_const = 0;
	t.is_pointer_volatile = 0;
	t.is_pointer_restrict = 0;
	return t;
}

internal void 
emit_exceptionhandling(const char *fn, CSerpentArgs flags, int n_fnargs, Symbol fnargs[])
{
	if(flags.error_handling.active) {
		
		if(flags.error_handling.fn) {
			assert(flags.error_handling.argno >= 0);

			if(flags.error_handling.argno > n_fnargs)
				die(0, "Error wrapping function '%s' in file '%s': "
				    "flag -e,%i,%s was specified, but function only has "
				    "%i arguments", 
				    fn, flags.filename, 
				    flags.error_handling.argno, flags.error_handling.fn,
				    n_fnargs);

			char *exnarg = flags.error_handling.argno == 0 
				? "rtn"
				: fnargs[flags.error_handling.argno-1].name;

			printf("    const char *_exn = %s(%s);  \n", flags.error_handling.fn, exnarg);
			printf("    if(_exn) {  \n");
			printf("        PyErr_SetString(PyExc_RuntimeError, _exn);  \n");
			printf("        return 0;  \n");
			printf("    }  \n");
		} else {
			assert(flags.error_handling.argno == 0);
	
			printf("    if(rtn) {  \n");
			printf("        PyErr_SetString(PyExc_RuntimeError, rtn);  \n");
			printf("        return 0;  \n");
			printf("    }  \n");
		}	
	}
}

internal void 
emit_call(const char *fn, CSerpentArgs flags, int n_fnargs, Symbol fnargs[])
{
	printf("%s (", fn);

	for(int i = 0; i < n_fnargs; i++)
	{
		char *sep  =  i ? ", " : "";
		if (is_array(fnargs[i].type))
			printf("%s%s_data", sep, fnargs[i].name);
		else 
			printf("%s%s", sep, fnargs[i].name);
	}

	printf(");\n");
}

internal int 
emit_py_buildvalue_fmt_char(Type t) 
{
	if      (t.category == T_CHAR) printf("b");
	else if (t.category == T_DOUBLE && !t.is_complex && !t.is_imaginary) printf("d");
	else if (t.category == T_FLOAT && !t.is_complex && !t.is_imaginary) printf("f");
	else if (t.category == T_SHORT && !t.is_unsigned) printf("h");
	else if (t.category == T_INT && !t.is_unsigned) printf("i");
	else if (t.category == T_LONG && !t.is_unsigned) printf("l");
	else if (t.category == T_LLONG && !t.is_unsigned) printf("L");
	else if (t.category == T_SHORT) printf("H");
	else if (t.category == T_INT) printf("I");
	else if (t.category == T_LONG) printf("k");
	else if (t.category == T_LLONG) printf("K");
	else if (t.category == T_BOOL) printf("p");
	else return 0;
	return 1;
}

internal void 
emit_wrapper (const char *fn, CSerpentArgs flags, int n_fnargs, Symbol fnargs[], Type rtntype)
{
	assert(n_fnargs >= 0);

	// declaration for function to be wrapped
	if(!flags.disable_declarations) {
		char buf[200] = {0};
		assert(ssizeof(buf) > repr_type(sizeof(buf), buf, rtntype));

		printf("%s %s (", buf, fn);

		if (n_fnargs == 0) 
			printf("void");

		else for(int i = 0; i < n_fnargs; i++)
		{
			memset(buf,0,sizeof(buf));
			assert(ssizeof(buf) > repr_type(sizeof(buf), buf, fnargs[i].type));

			char * sep  =  i ? ", " : "";
			printf("%s%s %s", sep, buf, fnargs[i].name);
		}


		printf(");\n");
	}

	// start of wrapper definition
	printf("PyObject * wrap_%s (PyObject *self, PyObject *args, PyObject *kwds)\n{\n",fn);
	printf("    (void) self;\n");

	if(n_fnargs) {

		// keyword name list
		printf("    static char *kwlist[] = {");
	        for(int i = 0; i < n_fnargs; i++)
			printf("\n        \"%s\",", fnargs[i].name);
		printf("0};\n");

		// declare a C variable for each argument
		for(int i = 0; i < n_fnargs; i++) {
			Symbol arg = fnargs[i];

			if (is_string(arg.type)) {
				printf("    char * %s = 0;\n", arg.name);
			}

			else if (is_voidptr(arg.type)) {
				printf("    unsigned long long %s_ull = 0;\n", arg.name);
				printf("    void * %s = 0;\n", arg.name);
			}

			else if (is_array(arg.type)) {
				printf("    PyObject *%s_obj = NULL;\n", arg.name);
				printf("    void *%s_data    = NULL;\n", arg.name);
			}

			else if (arg.type.category == T_BOOL) {
				printf("    int %s = 0;\n", arg.name);
			}

			else {
				char buf[200] = {0};
				assert(ssizeof(buf) > repr_type(sizeof(buf), buf, arg.type));
				printf("    %s %s = {0};\n", buf, arg.name);
			}
		}

		// parse python arguments into the above declared C variables
		printf("\n    if(!PyArg_ParseTupleAndKeywords(args, kwds, \"");
		for (int i = 0; i < n_fnargs; i++) {
			// building the format string for ParseTupleAndKeywords
			Symbol arg = fnargs[i];

			if      (is_string(arg.type))  printf("z");
			else if (is_voidptr(arg.type)) printf("K");
			else if (is_array(arg.type))   printf("O");
			else {
				Type t = arg.type;
				if (!emit_py_buildvalue_fmt_char(t)) {

					char buf[200] = {0};
					assert(ssizeof(buf) > repr_type(sizeof(buf), buf, t));
					die(0, "Error wrapping function '%s' in file '%s': "
					    "argument %i has type '%s', "
					    "which c-serpent doesn't know how to convert "
					    "from python",
					    fn, flags.filename, i, buf);
				}
			}
		}
		printf("\", kwlist");
		for (int i = 0; i < n_fnargs; i++) {
			// emit addresses for the arguments we actually want
			printf(",\n        ");
			Symbol arg = fnargs[i];

			if      (is_string(arg.type))  printf("&%s", arg.name);
			else if (is_voidptr(arg.type)) printf("&%s_ull", arg.name);
			else if (is_array(arg.type))   printf("&%s_obj", arg.name);
			else  printf("&%s", arg.name);

		}
		printf(")) return 0;\n\n");

		// type checking for any numpy arrays, conversions for any void pointers
		for (int i = 0; i < n_fnargs; i++)
		{
			Symbol arg = fnargs[i];

			if (is_voidptr(arg.type)) 
				printf("    memcpy(&%s, &%s_ull, sizeof(%s));\n", arg.name, arg.name, arg.name);

			else if (is_array(arg.type)) {
				char buf[200] = {0};
				assert(ssizeof(buf) > repr_type(sizeof(buf), buf, basetype(arg.type)));

				// emit array type check
				printf("    if (%s_obj != Py_None) { \n", arg.name);
				printf("        if (!PyArray_Check(%s_obj)) { \n", arg.name);
				printf("            PyErr_SetString(PyExc_ValueError, \"Argument '%s' must be a numpy array, or None\"); \n", arg.name);
				printf("            return 0; \n");
				printf("        } \n");
				printf("        if (PyArray_TYPE((PyArrayObject*)%s_obj) != C2NPY(%s)) {\n", arg.name, buf);
				printf("            PyErr_SetString(PyExc_ValueError, \"Invalid array data type for argument '%s' (expected %s)\");\n", arg.name, buf);
			        printf("            return 0; \n");
				printf("        } \n");

				// emit array contiguity check
				printf("        if(!PyArray_ISCARRAY((PyArrayObject*)%s_obj)) {\n", arg.name);
				printf("            PyErr_SetString(PyExc_ValueError, \"Argument '%s' is not C-contiguous\");\n", arg.name);
			        printf("            return 0;\n");
				printf("        }\n");
				printf("        %s_data = PyArray_DATA((PyArrayObject*)%s_obj); \n", arg.name, arg.name);
				printf("    }\n");
			}
		}


	}	
	else {
		printf("    (void) args;\n    (void) kwds;\n");
	}
	printf("\n");

	// now, emit the actual call

	if (is_plainvoid(rtntype)) {

		printf("    Py_BEGIN_ALLOW_THREADS;\n");
		printf("    ");
		emit_call(fn, flags, n_fnargs, fnargs);
		printf("    Py_END_ALLOW_THREADS;\n");
		emit_exceptionhandling(fn, flags, n_fnargs, fnargs);
		printf("    Py_RETURN_NONE;\n");

	} else if (is_string(rtntype)) {

		char buf[200] = {0};
		assert(ssizeof(buf) > repr_type(sizeof(buf), buf, rtntype));

		printf("    %s rtn = 0;\n", buf);
		printf("    Py_BEGIN_ALLOW_THREADS;\n");
		printf("    rtn = ");
		emit_call(fn, flags, n_fnargs, fnargs);
		printf("    Py_END_ALLOW_THREADS;\n");
		emit_exceptionhandling(fn, flags, n_fnargs, fnargs);
		printf("    return Py_BuildValue(\"s\", rtn);\n");

	} else if (is_voidptr(rtntype)) {

		char buf[200] = {0};
		assert(ssizeof(buf) > repr_type(sizeof(buf), buf, rtntype));

		printf("    %s rtn = 0;\n", buf);
		printf("    Py_BEGIN_ALLOW_THREADS;\n");
		printf("    rtn = ");
		emit_call(fn, flags, n_fnargs, fnargs);
		printf("    Py_END_ALLOW_THREADS;\n");
		emit_exceptionhandling(fn, flags, n_fnargs, fnargs);
		printf("    return PyLong_FromVoidPtr(rtn);\n");

	} else if (is_array(rtntype)) {

		char buf[200] = {0};
		assert(ssizeof(buf) > repr_type(sizeof(buf), buf, rtntype));

		die(0, "Error wrapping function '%s' in file '%s': "
		       "return type '%s' is not supported by c-serpent",
		       fn, flags.filename, buf);

	} else {
		char buf[200] = {0};
		assert(ssizeof(buf) > repr_type(sizeof(buf), buf, rtntype));

		// python requires us to handle bools as ints
		if (rtntype.category == T_BOOL) 
			printf("    int rtn = 0;\n");
		else
			printf("    %s rtn = 0;\n", buf);

		printf("    Py_BEGIN_ALLOW_THREADS;\n");
		printf("    rtn = ");
		emit_call(fn, flags, n_fnargs, fnargs);
		printf("    Py_END_ALLOW_THREADS;\n");
		emit_exceptionhandling(fn, flags, n_fnargs, fnargs);
		printf("    return Py_BuildValue(\"");
		if(!emit_py_buildvalue_fmt_char(rtntype)) {
			die(0, "Error wrapping function '%s' in file '%s': "
			       "return type '%s' is not supported by c-serpent",
			       fn, flags.filename, buf);
		}
		printf("\", rtn);\n");
	}

	printf("}\n\n");

}

internal void 
emit_dispatch_wrapper (
	ParseCtx p,
	const char *fn, 
	int n_variants_implemented,
	short arg_match_count[static MAX_FN_ARGS],
	int n_supported_suffixes,
	VariantSuffix suffixes[],
	Symbol fnargs[static MAX_FN_ARGS] ) 
{
	// emit a very general wrapper that just dispatches based on the type of the first argument that matches the suffix in all implementations 
	
	int idx_first_covariant_arg = -1;
	int n_args = 0;

	// try to find a covariant array argument first, only use scalar as a backup
	for (int i = MAX_FN_ARGS-1; i >= 0; i--) {
		if (arg_match_count[i] == n_variants_implemented && fnargs[i].type.is_pointer) 
			idx_first_covariant_arg = i;
		if (fnargs[i].type.category != T_UNINITIALIZED && fnargs[i].type.category != T_UNKNOWN)
			n_args++;
	}

	if (idx_first_covariant_arg < 0) 
		for (int i = MAX_FN_ARGS-1; i >= 0; i--) 
			if (arg_match_count[i] == n_variants_implemented) 
				idx_first_covariant_arg = i;

	// but there does need to be at least ONE covariant arg... 
	if (idx_first_covariant_arg < 0) 
		die(&p, "couldn't generate generic wrapper for '%s', no argument correctly matches suffix in all implemented variants.", fn);

	Type key_arg = fnargs[idx_first_covariant_arg].type;

	printf("PyObject * wrap_%s (PyObject *self, PyObject *args, PyObject *kwds)\n{\n",fn);

	printf("    PyObject *arglist[%i] = {0};\n", n_args);

	// keyword name list
	printf("    static char *kwlist[] = {");
	for(int i = 0; i < n_args; i++)
		printf("\n        \"%s\",", fnargs[i].name);
	printf("0};\n");

	// extract args
	printf("    if(!PyArg_ParseTupleAndKeywords(args, kwds, \"" );
	for (int i = 0; i < n_args; i++) printf("O");
	printf("\", kwlist");
	for (int i = 0; i < n_args; i++) printf(",&arglist[%i]", i);
	printf(")) return 0;\n");

	// emit dispatch if statements
	for (int i = 0; i < n_supported_suffixes; i++) {

		if(suffixes[i].found) {

			if(key_arg.is_pointer) {

				char buf[200] = {0};
				assert(ssizeof(buf) > repr_type(sizeof(buf), buf, basetype(suffixes[i].type)));
				// array 
				printf("    if (PyArray_Check(arglist[%i]) && PyArray_TYPE((PyArrayObject*)arglist[%i]) == C2NPY(%s)) {\n", idx_first_covariant_arg, idx_first_covariant_arg, buf);

				printf("        return wrap_%s%c(self, args, kwds);\n", fn, suffixes[i].suffix);

				printf("    } else ");

			} else {

				// non-array

				// PyLong
				if (suffixes[i].type.category >= T_CHAR && suffixes[i].type.category <= T_LLONG) {
					printf("    if (PyLong_Check(arglist[%i])) {\n", idx_first_covariant_arg);
					printf("        return wrap_%s%c(self, args, kwds);\n", fn, suffixes[i].suffix);
					printf("    } else ");
				}

				// PyFloat / PyComplex
				else if (suffixes[i].type.category >= T_FLOAT && suffixes[i].type.category <= T_LDOUBLE) {

					if (suffixes[i].type.is_complex) {
						printf("    if (PyComplex_Check(arglist[%i])) {\n", idx_first_covariant_arg);
						printf("        return wrap_%s%c(self, args, kwds);\n", fn, suffixes[i].suffix);
						printf("    } else ");
					} else {
						printf("    if (PyFloat_Check(arglist[%i])) {\n", idx_first_covariant_arg);
						printf("        return wrap_%s%c(self, args, kwds);\n", fn, suffixes[i].suffix);
						printf("    } else ");
					}
				}

				// ..wat?
				else {
					char buf[200] = {0};
					assert(ssizeof(buf) > repr_type(sizeof(buf), buf, key_arg));
					die(&p, "error generating dispatcher for '%s' unsupported scalar argument type '%s'", fn, buf);
				}
				
			}
		}
	}

	printf("{\n        PyErr_SetString(PyExc_ValueError, \"No instance of generic function '%s' matches supplied argument types\");\n        return 0;\n    }\n", fn);

	printf("}\n");
}

/*
	==========================================================
		Parsing
	==========================================================
*/


internal int 
eat_identifier(ParseCtx *p, const char *id)
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

internal int 
eat_token(ParseCtx *p, long toktype)
{
	if(p->tokens == p->tokens_end) return 0;
	if(p->tokens[0].toktype == toktype) {
		p->tokens++;
		return 1;
	}
	return 0;
}

internal int
check_token_is_identifier(Token *t, const char *id, long id_len)
{
	if (id_len == 0) 
		id_len = strlen(id);

	if (t->toktype == CLEX_id
		&& t->string_len == id_len
		&& !memcmp(t->string, id, id_len)) {
		return 1;
	}

	return 0;
}

internal int 
identifier(ParseCtx *p, char** out_id)
{
	if(p->tokens == p->tokens_end) return 0;
	if(p->tokens[0].toktype == CLEX_id) {
		if(out_id) 
			*out_id = p->tokens[0].string;
		p->tokens++;
		return 1;
	}
	return 0;
}

internal int 
typedef_name(ParseCtx *p, Type *t)
{
	if(p->tokens == p->tokens_end) return 0;
	if(p->tokens[0].toktype == CLEX_id) {

		Symbol *s = 0;
		if ((s = get_symbol(p->storage, p->tokens[0].string))) {
			if(t) *t = s->type;
			p->tokens++;
			return 1;
		}
	}
	return 0;
}

internal int 
supported_type(ParseCtx *p, Type *t)
{
	if (p->tokens == p->tokens_end) return 0;

	else if (typedef_name(p, t)) return 1;

	else if (eat_token(p, '*')) { return modify_type_pointer(p,t); }

	else if (p->tokens[0].toktype == CLEX_id) {

		if (eat_identifier(p, "void"))     { modify_type_void(p,t); return 1; }
		if (eat_identifier(p, "char"))     { modify_type_char(p,t); return 1; } 
		if (eat_identifier(p, "short"))    { modify_type_short(p,t); return 1; }
		if (eat_identifier(p, "int"))      { modify_type_int(p,t); return 1; }
		if (eat_identifier(p, "long"))     { modify_type_long(p,t); return 1; }
		if (eat_identifier(p, "float"))    { modify_type_float(p,t); return 1; }
		if (eat_identifier(p, "double"))   { modify_type_double(p,t); return 1; }
		if (eat_identifier(p, "signed"))   { modify_type_signed(p,t); return 1; }
		if (eat_identifier(p, "unsigned")) { modify_type_unsigned(p,t); return 1; }
		if (eat_identifier(p, "_Bool"))      { modify_type_bool(p,t); return 1; }
		if (eat_identifier(p, "_Complex"))    { modify_type_complex(p,t); return 1; }
		if (eat_identifier(p, "_Imaginary"))  { modify_type_imaginary(p,t); return 1; }

		if (eat_identifier(p, "const"))  { modify_type_const(p,t); return 1; }
		if (eat_identifier(p, "restrict"))  { modify_type_restrict(p,t); return 1; }
		if (eat_identifier(p, "volatile"))  { modify_type_volatile(p,t); return 1; }

		return 0;
	}
	return 0;	
}

internal int 
supported_type_list(ParseCtx *p, Type *t)
{
	if(!supported_type(p,t)) return 0;
	while(supported_type(p,t)) {}
	return 1;
}

internal int 
supported_typedef(ParseCtx *p, Symbol *s)
{
	SAVE(p);
	Symbol tmp = *s;

	if(eat_identifier(p, "typedef")
		&& supported_type_list(p, &tmp.type)
		&& identifier(p, &tmp.name)
		&& eat_token(p, ';')) 
	{ 
		// can these actually happen in valid code? don't think so.. 
		if(!strcmp(tmp.name, "struct")) { RESTORE(p); return 0; }
		if(!strcmp(tmp.name, "union"))  { RESTORE(p); return 0; }
		if(!strcmp(tmp.name, "enum"))   { RESTORE(p); return 0; }
		*s = tmp;
		return 1; 
	}

	RESTORE(p);
	return 0;
}

internal void 
populate_symbols(StorageBuffers *storage, ParseCtx p)
{
	while(p.tokens != p.tokens_end) {

		while (!check_token_is_identifier(p.tokens, "typedef", 7)) 
		{ 
			p.tokens++;
			if (p.tokens == p.tokens_end) return;
		}

		Symbol s = {0};
		if (supported_typedef(&p, &s)) {
			add_symbol(storage, s);

			if(p.args.verbose) {
				char buf[200] = {0};
				repr_symbol(sizeof(buf), buf, s);
				fprintf(stderr, "registered type %s\n", buf); 
			}

		} else {
			p.tokens++;
		}
	}
}


internal int 
arg(ParseCtx *p, const char *fn, Symbol *fnarg, int fatal)
{
	SAVE(p);

	Symbol tmp = {0};

	if(!supported_type_list(p, &tmp.type)) {
		if(fatal) die(p, "error wrapping function '%s' in '%s': unsupported type", fn, p->args.filename);
		RESTORE(p);
		return 0;
	}

	if(!identifier(p, &tmp.name)) {
		if(fatal) die(p, "error wrapping function '%s' in '%s': expected identifier (i.e. argument name)", fn, p->args.filename);
		RESTORE(p);
		return 0;
	}

	if(eat_token(p,'[')) {
		if(!modify_type_pointer(p, &tmp.type)) {
			if(fatal) die(p, "error wrapping function '%s' in '%s': unsupported type", fn, p->args.filename);
			return 0;
		}
		while(1) {
			if (eat_identifier(p, "static")) {}
			else if (eat_identifier(p, "const")) {modify_type_const(p, &tmp.type);}
			else if (eat_identifier(p, "restrict")) {modify_type_restrict(p, &tmp.type);}
			else break;
		}

		// skip over everything else until we close the square bracket
		// this isn't robust, just a hack that will probably work on common valid C code
		int depth = 1;
		while(depth > 0)
		{
			if(eat_token(p, '[')) depth++;
			else if(eat_token(p, ']')) depth--;
			else {
				p->tokens++;
				if(p->tokens == p->tokens_end) 
					die(p, "parse error: unexpected end of file");
			}
		}
	}

	*fnarg = tmp;
	return 1;
}

internal int 
arglist(ParseCtx *p, const char *fn, int max_args, int *num_args, Symbol fnargs[], int fatal)
{
	SAVE(p);
	*num_args = 0;

	if (eat_token(p, '(') 
		&& eat_identifier(p, "void") 
		&& eat_token(p, ')')) {

		return 1;
	}

	RESTORE(p);

	if (eat_token(p, '('))
	{
		while(arg(p, fn, fnargs+(*num_args), fatal)) 
		{
			*num_args = *num_args + 1;
			if(eat_token(p, ',') && (*num_args == max_args))
				die(p, "error wrapping function '%s' in '%s': functions with more than %i arguments are not supported", fn, p->args.filename, max_args);
		}

		if(!eat_token(p, ')')) {
			RESTORE(p);
			return 0;
		}

		return 1;
	}

	RESTORE(p);
	return 0;
}

internal void 
attributes(ParseCtx *p, const char *fn)
{
	while(1) {
		if (eat_identifier(p, "__attribute__")) {
			if (!eat_token(p, '(')) {
				die(p, "error wrapping function '%s' in '%s': parse error (malformed attribute)", fn, p->args.filename);
			}

			// skip over content of the brackets
			// not super robust
			int depth = 1;
			while(depth > 0) {
				if(eat_token(p, '(')) depth++;
				else if(eat_token(p, ')')) depth--;
				else {
					p->tokens++;
					if(p->tokens == p->tokens_end) 
						die(p, "parse error: unexpected end of file");
				}
			}
		}
		else break;
	}
}

internal void 
process_function(ParseCtx p, Symbol argsyms[static MAX_FN_ARGS])
{
	// on entry, p.tokens is set right on the function name.	
	const char * fn = p.tokens[0].string;
	// rewind to semicolon
	while(p.tokens[0].toktype != ';') p.tokens--;
	p.tokens++;

	Type rtn_t = {0};
	memset(argsyms, 0, MAX_FN_ARGS*sizeof(argsyms[0]));

	if(!supported_type_list(&p, &rtn_t))
		die(&p, "error wrapping function '%s' in '%s': unsupported return type", fn, p.args.filename);
	
	char *sanity_check = 0;
	if(!identifier(&p, &sanity_check)) 
		die(&p, "error wrapping function '%s' in '%s': unsupported specifiers or qualifiers", fn, p.args.filename);
	if(sanity_check != fn)
		die(&p, "error wrapping function '%s' in '%s': parse error (encountered unrecognized garbage)", fn, p.args.filename);

	int num_args = 0;
	if(!arglist(&p, fn, MAX_FN_ARGS, &num_args, argsyms, 0)) {
		arglist(&p, fn, MAX_FN_ARGS, &num_args, argsyms, 1);
		die(&p, "error wrapping function '%s' in '%s': parse error (couldn't parse argument list)", fn, p.args.filename);
	}

	attributes(&p, fn);

	if(!eat_token(&p, ';'))
		die(&p, "error wrapping function '%s' in '%s': parse error (encountered unrecognized garbage)", fn, p.args.filename);

	// Parse successful, emit the wrapper!
	emit_wrapper (fn, p.args, num_args, argsyms, rtn_t);
}

internal int 
parse_file(ParseCtx p, const char *function_name, Symbol argsyms[static MAX_FN_ARGS])
{
	long len = strlen(function_name);

	while (p.tokens != p.tokens_end) {
		if (check_token_is_identifier(p.tokens, function_name, len) )
		{
			process_function(p, argsyms);
			return 1;
		}
		p.tokens++;
	}
	return 0;
}


/*
	==========================================================
		Lexing
	==========================================================
*/

#ifdef _WIN32
#define popen(x,y) _popen(x,y)
#define pclose(x) _pclose(x)
#endif


internal void 
print_context(char *start, char *loc)
{
	char *first = loc;
	while(loc-first < 60 && first >= start) 
		first--;
	while(*first != '\n' && first >= start) 
		first--;

	char *last = loc;
	while(last-loc < 60 && *last != 0) 
		last++;
	while(*last != '\n' && *last != 0) 
		last++;

	while(first != last) {
		if (first == loc) fprintf(stderr, " HERE>>>");
		putc(*(first++),stderr);
	}
}

enum delim_stack_action {
	DS_PUSH,
	DS_POP,
	DS_QUERY,
};

internal long 
delim_stack(enum delim_stack_action action, Token value, char *start, char* loc) {
	// TODO not thread safe
	global short stack[200] = {0};
	global char *locations[200] = {0};
	global long pos = 0;

	switch(action) {
	case DS_PUSH:
		if(pos == COUNT_ARRAY(stack)) die(0, "congratulations, your file blew the delimiter stack");
		locations[pos] = loc;
		stack[pos++] = value.toktype;
		return -1;
	case DS_POP:
		if(pos == 0) {
			fprintf(stderr, "mismatched delimiters (extraneous %c)\n\ncontext:\n", (char)value.toktype);
			print_context(start, loc);
			fputc('\n', stderr);
			exit(EXIT_FAILURE);
		}
		switch(value.toktype)
		{
		case '}':
			if ('{' != stack[--pos]) {
				fprintf(stderr, "mismatched delimiters (got '}' to match '%c')\n\n", stack[pos]);
				goto mismatch_close;
			}
			return delim_stack(DS_QUERY, value, start, loc);
		case ')':
			if ('(' != stack[--pos]) {
				fprintf(stderr, "mismatched delimiters (got ')' to match '%c')\n\n", stack[pos]);
				goto mismatch_close;
			}
			break;
		case ']':
			if ('[' != stack[--pos]) {
				fprintf(stderr, "mismatched delimiters (got ']' to match '%c')\n\n", stack[pos]);
				goto mismatch_close;
			}
			break;
		default:
			assert(0);
		}
		return 0;
	case DS_QUERY:
		for (int i = 0; i < pos; i++)
			if(stack[i] == '{') return 0;
		return 1;
	default: 
		assert(0);
	}
	return -1;	

mismatch_close:
	fprintf(stderr, "opening delimiter context:\n\n");
	print_context(start, locations[pos]);
	fprintf(stderr, "\n\ninvalid closing delimiter context:\n\n");
	print_context(start, loc);
	fputc('\n', stderr);
	exit(EXIT_FAILURE);
}

internal void delim_push(Token value, char *start, char* loc) { (void)delim_stack(DS_PUSH, value, start, loc); }
internal int  delim_pop(Token value, char *start, char* loc) { return delim_stack(DS_POP, value, start, loc); }
internal int  toplevel(void) { return delim_stack(DS_QUERY, (Token){0}, 0, 0); }


internal int 
lex_file(StorageBuffers *st, CSerpentArgs args, long long tokens_maxnum, Token *tokens, long long text_bufsz, char *text, long long string_store_bufsz, char *string_store)
{
	int ntok = 0;

	// preserve the start of the text buffer and ensure always null terminated.
	char *text_start = text;
	memset(text,0,text_bufsz);
	text_bufsz--;

	/* 
		If we're manually including files, do so.
	*/

	for (int i = 0; i < args.manual_include.nfiles; i++)
	{
		// try the current directory
		FILE *f = fopen(args.manual_include.files[i], "rb");

		// try the list of -I paths
		if(!f) for (int j = 0; j < args.ndirs; j++)
		{
			char path[4096] = {0};
			// TODO accomodate windows
			if(ssizeof(path) <= snprintf(path, sizeof(path), "%s/%s", args.dirs[j], args.manual_include.files[i])) 
				die(0, "internal buffer overflow (path too long)");
			f = fopen(path, "rb");
			if(f) break;
		}

		// try the system default path
		if(!f) {
			char path[4096] = {0};
			// TODO accomodate windows
			if(ssizeof(path) <= snprintf(path, sizeof(path), "/usr/include/%s", args.manual_include.files[i])) 
				die(0, "internal buffer overflow (path too long)");
			f = fopen(path, "rb");
		}

		if(!f) die(0, "couldn't find file to be manually included: %s", args.manual_include.files[i]);

		long long len = fread(text, 1, text_bufsz, f);
		if(len == text_bufsz) die(0,"input file too long");
		text += len;
		text_bufsz -= len;

		fclose(f);
	}

	/*
		Read input file
	*/

	if(!args.preprocessor || args.disable_pp) {

		// read the usual way
		FILE *f = fopen(args.filename, "rb");
		if(!f) die(0, "couldn't fopen '%s'", args.filename);
		long long len = fread(text, 1, text_bufsz, f);
		if(len == text_bufsz) die(0,"input file too long");
		fclose(f);

	} else {

		// read via popen to preprocessor command
		char cmd[4096] = {0};
		if(ssizeof(cmd) <= snprintf(cmd, sizeof(cmd), "%s %s", args.preprocessor, args.filename)) 
			die(0, "internal buffer overflow");
		
		FILE *f = popen(cmd, "r");
		if(!f) die(0, "couldn't popen '%s'", cmd);
		long long len = fread(text, 1, text_bufsz, f);
		if(len == text_bufsz) die(0,"input file too long");
		int exit_status = pclose(f);
		switch (exit_status) {
			case  0: break;
			case -1: die(0, "wait4 on '%s' failed, or other internal error occurred", cmd);
			default: die(0, "'%s' failed with code %i", cmd, exit_status);
		}
	}
	
	/*
		Lex whole file
	*/

	tokens[ntok++] = (Token){.toktype=';'};

	stb_lexer lex = {0};
	stb_c_lexer_init(&lex, text_start, text+strlen(text_start), (char *) string_store, string_store_bufsz);
	while(stb_c_lexer_get_token(&lex)) {
		if(tokens_maxnum == ntok) die(0, "internal buffer overflow");

		Token t = {.toktype = lex.token};
		switch(lex.token)
		{
			case CLEX_eof:
				t.toktype = ';';
				break;
			case CLEX_id: 
			case CLEX_dqstring:
			case CLEX_sqstring: 		
				t.string_len = strlen(lex.string);	
				t.string = intern_string(st, lex.string, t.string_len);
				break;
			case CLEX_charlit:
				t.int_number = lex.int_number;
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

		// skip all braced code.
		// when scanning braced code, check that delimiters match, but that's it. 

		if(t.toktype == '{' || t.toktype == '(' || t.toktype == '[') {
			delim_push(t, text_start, lex.where_firstchar);
		} else if(t.toktype == '}' || t.toktype == ')' || t.toktype == ']') {
			if(delim_pop(t, text_start, lex.where_firstchar)) {
				tokens[ntok++] = (Token){.toktype=';'};
				continue;
			}
		}

		if(toplevel()) {
			tokens[ntok++] = t;
		} 
	}

	return ntok;
}


/*
	==========================================================
		Main
	==========================================================
*/


internal void 
usage(void)
{
	const char *message = 

	"c-serpent \n"
	"========= \n"
	"                                                                             \n"
	"Typical usage:  \n"
	" $ c-serpent -m coolmodule -f my_c_file.c function1 function2 > wrappers.c   \n"
	" $ cc -fPIC -shared -I/path/to/python/headers \\\n"
	"       wrappers.c my_c_file.c \\\n"
	"       -lpython -o coolmodule.so\n"
	"                                                                             \n"

	"C-serpent processes its input arguments in-order. First specify the name of the  \n"
	"output python module (which must match the name of the shared library that  \n"
	"you compile) by using the argument sequence '-m modulename'. Then, specify  \n"
	"at least one file, using '-f filename.c'. Then, list the names of the   \n"
	"functions that you wish to generate wrappers for. You can specify multiple  \n"
	"files like so: '-f minmax.c min max -f avg.c mean median'. The functions are  \n"
	"assumed to be contained in the file specified by the most recent '-f' flag.  \n"
	"  \n"
	"C-serpent invokes the system preprocessor and scans for typedefs in the   \n"
	"resulting file. It only understands a subset of all possible C typedefs, but  \n"
	"it works for stdint, size_t, and so on. The preprocessor to use is 'cc -E'   \n"
	"by default, but this can be overridden with the -p flag, or the CSERPENT_PP  \n"
	"environment variable (the latter takes precedence if both are supplied).  \n"
	"  \n"
	"Flags:   \n"
	"                                                                               \n"
	"-h   print help message and exit    \n"
	"  \n"
	"-m   the following argument is the name of the module to be built   \n"
	"     only one module per c-serpent invocation is allowed.  \n"
	"                                                                               \n"
	"-f   the following argument is a filename.  \n"
	"                                                                               \n"
	"-v   verbose (prints a list of typedefs that were parsed, for debugging).  \n"
	"                                                                               \n"
	"-D   disable including declarations for the functions to be wrapped in the   \n"
	"     generated wrapper file this might be used to facilitate amalgamation   \n"
	"     builds, for example.  \n"
	"                                                                                                                                                           \n"
	"-x   if you have some extra handwritten wrappers, you can use '-x whatever'    \n"
	"     to include the function 'whatever' (calling 'wrap_whatever') in the       \n"
	"     generated module. You'll need to prepend the necessary code to the file   \n"
	"     that c-serpent generates.  \n"
	"                                                                               \n"
	"-p   the following argument specifies the preprocessor to use for future   \n"
	"     files, if different from the default 'cc -E'. Use quotes if you need  \n"
	"     to include spaces in the preprocessor command.  \n"
	"                                                                               \n"
	"-P   disable preprocessing of the next file encountered. This flag only lasts   \n"
	"     until the next file change (i.e. -f).  \n"
	"                                                                               \n"
	"-t   the following argument is a type name, which should be treated as being \n"
	"     equivalent to void. This is useful for making c-serpent handle pointers to \n"
	"     unsupported types (e.g. structs) as void pointers (thereby converting them \n"
	"     to and from python integers). \n"
	"                                                                               \n"
	"     this flag only lasts until the next file change (i.e. -f)   \n"
	"                                                                               \n"
	"-i   the following argument is a filename, to be inlcuded before the next    \n"
	"     file processed (for use with -P).  \n"
	"                                                                               \n"
	"-I   the following argument is a directory path, to be searched for any    \n"
	"     future -i flags.  \n"
	"                                                                               \n"
	"-g   functions that follow are \"generic\". This is explained fully below.      \n"
	"                                                                               \n"
	"     this flag only lasts until the next file change (i.e. -f)   \n"
	"                                                                               \n"
	"-G   By default, when processing generic functions, c-serpent will remove a  \n"
        "     trailing underscore from the names of the generated dispatcher function \n"
	"     (e.g. for functions sum_f and sum_d, the arguments -g -G sum_ would result\n"
	"     in the dispatcher function simply being called sum). This flag disables \n"
	"     that functionality, causing trailing underscores to be kept.            \n"
	"                                                                               \n"
	"     this flag only lasts until the next file change (i.e. -f)   \n"
	"                                                                               \n"
	"-e   for functions that follow: if they return a string (const char *), the    \n"
	"     string is to be interpreted as an error message (if not null) and a python  \n"
	"     exception should be thrown.  \n"
	"                                                                               \n"
	"     this flag only lasts until the next file change (i.e. -f)   \n"
	"                                                                               \n"
	"-e,n,chkfn   for functions that follow: after calling, another function called  \n"
	"     chkfn should be called.  chkfn should have the signature    \n"
	"     'const char * checkfn (?)' where ? is the type of the n-th argument to the  \n"
	"     function (0 means the function's return value). if the chkfn call returns  \n"
	"     a non-null string, that string is assumed to be an error message and a    \n"
	"     python exception is generated.   \n"
	"                                                                               \n"
	"     this flag only lasts until the next file change (i.e. -f)   \n"
	"                                                                               \n"
	"Environment variables:   \n"
	"                                                                               \n"
	"CSERPENT_PP    \n"
	"     This variable acts like the -p flag (but the -p flag overrides it)   \n"

	"                                                                               \n"
	"Generic functions:   \n"
	"                                                                               \n"
	"     If you have several copies of a function that accept arguments that are   \n"
	"     of different data types, then c-serpent may be able to automatically      \n"
	"     generate a disapatch function for you, that allows it to be called from   \n"
	"     python in a type-generic way. In order to use this feature, your function \n"
	"     must use a function-name suffix to indicate the data type, following this \n"
	"     convention: \n"
	"                                                                               \n"
        "       type            suffix \n"
        "       ----            ------ \n"
        "       int8            b \n"
        "       int16           s \n"
        "       int32           i \n"
        "       int64           l \n"
        "        \n"
        "       uint8           B \n"
        "       uint16          S \n"
        "       uint32          I \n"
        "       uint64          L \n"
        "        \n"
        "       float           f \n"
        "       double          d \n"
        "        \n"
        "       complex float   F \n"
        "       complex double  D \n"
	"                                                                               \n"
	"     You do not need to supply all of these variants; c-serpent will support   \n"
	"     whichever variants it finds. \n"
	"                                                                               \n"
	"     Example: consider: $ ./c-serpent -m mymodule -f whatever.c -g mean        \n"
	"     If whatever.c contains the following functions, then python code will be  \n"
	"     able to call `mymodule.mean(N, arr)` where arr is a float or double array \n"
	"                                                                               \n"
	"       double meanf(int N, float *arr);                                        \n"
	"       double meand(int N, double *arr);                                       \n"
	"                                                                               \n"
	"     C-serpent will try to figure out which arguments change according to the  \n"
	"     convention and which do not. Return values may also change.               \n"
	"                                                                               \n"
	"     Lastly, the type-specific versions of the function do still get wrapped.  \n"
	;

	fprintf(stderr, "%s", message);
	
}


int 
cserpent_main (char *argv[])
{
	/*
		Allocate buffers
	*/
	
	char  *text         = calloc(1, 1<<27);
	char  *string_store = calloc(1, 0x10000);
	Token *tokens       = calloc(1, (1<<27) * sizeof(*tokens));
	StorageBuffers *storage = calloc(1, sizeof(*storage)); 

	if(!(text && string_store && tokens && storage)) 
		die(0,"out of mem");

	int emitted_preamble = 0;

	CSerpentArgs args = {.preprocessor = "cc -E"};
	if (getenv("CSERPENT_PP")) 
		args.preprocessor = getenv("CSERPENT_PP");

	const char *fnames[200] = {0};
	_Bool fname_needs_remove_underscore[200] = {0};
	int n_fnames = 0;

	int ntok = 0;

	while (*argv) {
		if (!strcmp(*argv, "-h")) {
			usage();
			exit(EXIT_SUCCESS);
		}

		if (!strcmp(*argv, "-v")) {
			args.verbose = 1;
			argv++;
			continue;
		}

		if (!strcmp(*argv, "-g")) {
			args.generic = 1;
			argv++;
			continue;
		}

		if (!strcmp(*argv, "-G")) {
			args.generic_keep_trailing_underscore = 1;
			argv++;
			continue;
		}

		if (!strcmp(*argv, "-P")) {
			args.disable_pp = 1;
			argv++;
			continue;
		}

		if (!strcmp(*argv, "-D")) {
			args.disable_declarations = 1;
			argv++;
			continue;
		}

		if (!strcmp(*argv, "-I")) { 
			argv++;
			if(*argv && **argv != '-') {
				if(args.ndirs == COUNT_ARRAY(args.dirs)) die(0, "A maximum of %i -I flags can be used in a single c-serpent invocation.", (int)COUNT_ARRAY(args.dirs));
				args.dirs[args.ndirs++] = *argv;
				argv++;
			} else {
				die(0, "-I flag must be followed by a directory (to be searched for files manually included with -i)");
			}
			continue;
		}

		if (!strcmp(*argv, "-i")) { 
			argv++;
			if(*argv && **argv != '-') {
				if(args.manual_include.nfiles == COUNT_ARRAY(args.manual_include.files)) die(0, "A maximum of %i -i flags can be used per file to be processed.", (int)COUNT_ARRAY(args.manual_include.files));
				args.manual_include.files[args.manual_include.nfiles++] = *argv;
				argv++;
			} else {
				die(0, "-i flag must be followed by a file path (to be manually included when processing the next file)");
			}
			continue;
		}
		
		if (!strcmp(*argv, "-m")) { 
			argv++;
			if(*argv && **argv != '-') {
				args.modulename = *argv;
				argv++;
			} else {
				die(0, "-m flag must be followed by a valid name (for the generated python module)");
			}
			continue;
		}
	
		if (!strcmp(*argv, "-x")) { 
			argv++;
			if(*argv && **argv != '-') {
				fnames[n_fnames++] = *argv;
				if(n_fnames == COUNT_ARRAY(fnames)) 
					die(0, "error: c-serpent only supports wrapping up to  %i functions", (int) COUNT_ARRAY(fnames));
				argv++;
			} else {
				die(0, "-x flag must be followed by a function name");
			}
			continue;
		}


		if (!strcmp(*argv, "-p")) { 
			argv++;
			if(*argv && **argv != '-') {
				args.preprocessor = *argv;
				argv++;
			} else {
				die(0, "-p flag must be followed by a program (to serve as preprocessor)");
			}
			continue;
		}

		if (!strcmp(*argv, "-t")) { 
			argv++;
			if(*argv && **argv != '-') {
				Symbol newtype = {
					.name = *argv,
					.type = {.category = T_VOID},
				};
				(void)add_symbol(storage, newtype);	
				argv++;
			} else {
				die(0, "-t flag must be followed by a type name");
			}
			continue;
		}

		if (!strcmp(*argv, "-f")) {
			argv++;

			if(!emitted_preamble) {
				emit_preamble(args); 
				emitted_preamble = 1;
			}

			if(*argv && **argv != '-') {
				memset(&args.error_handling, 0, sizeof(args.error_handling));
				args.filename = *argv;
				ntok = lex_file(storage, args, 1<<27, tokens, 1<<27, text, 0x10000, string_store );
				clear_symbols(storage);
				populate_symbols(
					storage, 
					(ParseCtx) {
						.tokens_first  =  tokens,
						.tokens        =  tokens,
						.tokens_end    =  tokens+ntok,
						.storage       =  storage,
						.args          =  args, }
					);
				args.disable_pp = 0;
				args.generic = 0;
				args.generic_keep_trailing_underscore = 0;
				memset(&args.manual_include, 0, sizeof(args.manual_include));
				argv++;
			} else {
				die(0, "-f flag must be followed by a filename");
			}
			continue;
		}

		if (!strncmp(*argv, "-e,", 3)) {
			char *c = *argv+3;
			int nchars_read = 0;
			int argno = xatoi(c, &nchars_read);
			c += nchars_read+1;
			if(c[-1] != ',') die(0,"expected comma in argument '%s'", *argv);
			args.error_handling.active = 1;
			args.error_handling.argno = argno;
			args.error_handling.fn = c;
			argv++;
			continue;
		}

		if (!strncmp(*argv, "-e", 2)) {
			args.error_handling.active = 1;
			args.error_handling.argno = 0;
			args.error_handling.fn = 0;
			argv++;
			continue;	
		}

		if (**argv == '-') {
			fprintf(stderr, "unrecognized flag: '%s'\n\n", *argv);
			usage();
			exit(EXIT_FAILURE);
		}

		/*
			Run parser on token array.
		*/

		fnames[n_fnames++] = *argv;
		if(n_fnames == COUNT_ARRAY(fnames)) 
			die(0, "error: c-serpent only supports wrapping up to  %i functions (including generic variants)", (int) COUNT_ARRAY(fnames));

		ParseCtx p = {
			.tokens_first  =  tokens,
			.tokens        =  tokens,
			.tokens_end    =  tokens+ntok,
			.args          =  args,
			.storage       =  storage,
		};

		Symbol argsyms[MAX_FN_ARGS] = {0};

		if(args.generic) {

			if ((*argv)[strlen(*argv)-1] == '_'  &&  !args.generic_keep_trailing_underscore) 
				fname_needs_remove_underscore[n_fnames-1] = 1;

			int n_variants_found = 0;
			VariantSuffix variant_suffixes[] = {

				/*
					The order is important here.
					For scalar arguments, the generated dispatcher will call
					the first version that appears.
				*/

				{get_symbol_or_die(storage, "int64_t")->type, 'l'},
				{get_symbol_or_die(storage, "int32_t")->type, 'i'},
				{get_symbol_or_die(storage, "int16_t")->type, 's'},
				{get_symbol_or_die(storage, "int8_t")->type,  'b'},

				{get_symbol_or_die(storage, "uint64_t")->type, 'L'},
				{get_symbol_or_die(storage, "uint32_t")->type, 'I'},
				{get_symbol_or_die(storage, "uint16_t")->type, 'S'},
				{get_symbol_or_die(storage, "uint8_t")->type,  'B'},

				{(Type){.category=T_DOUBLE}, 'd'},
				{(Type){.category=T_FLOAT},  'f'},

				{(Type){.category=T_DOUBLE, .is_complex=1}, 'D'},
				{(Type){.category=T_FLOAT, .is_complex=1},  'F'},
			};


			short arg_match_count[MAX_FN_ARGS] = {0}; 

			for (int i = 0; i < COUNT_ARRAY(variant_suffixes); i++) {

				char namebuf[500] = {0};	
				snprintf(namebuf, sizeof(namebuf), "%s%c", *argv, variant_suffixes[i].suffix);

				if (parse_file(p, namebuf, argsyms)) {

					fnames[n_fnames++] = intern_string(storage, namebuf, 0) ;
					if(n_fnames == COUNT_ARRAY(fnames)) 
						die(0, "error: c-serpent only supports wrapping up to  %i functions (including generic variants)", (int) COUNT_ARRAY(fnames));

					variant_suffixes[i].found = 1;
					n_variants_found++;

					for (int j = 0; j < MAX_FN_ARGS; j++) {
						arg_match_count[j] += compare_types_equal(argsyms[j].type, variant_suffixes[i].type, 0,0,0,0);
					}
				}
			}

			if (n_variants_found == 0) 
				die(0, "didn't find any variants of a function called '%s' in file '%s' that followed the required suffix convention", *argv, p.args.filename);

			emit_dispatch_wrapper(p, *argv, n_variants_found, arg_match_count, COUNT_ARRAY(variant_suffixes), variant_suffixes, argsyms);


		} else {

			if(!parse_file(p, *argv, argsyms)) 
				die(0, "didn't find function called '%s' in file '%s'", *argv, p.args.filename);
		}

		argv++;
	}

	emit_module(args, n_fnames, fnames, fname_needs_remove_underscore);
	return 0;
}

int 
main (int argc, char *argv[])
{
	(void)argc;
	argv++;
	if(!*argv) usage();
	return cserpent_main(argv);
}
