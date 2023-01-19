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
	const char *dirs[40];
	int ndirs;

	struct {
		const char *fn;
		short active;
	 	short argno;
	} error_handling;

	struct {
		const char *files[40];
		int nfiles;
	} manual_include;

} WrapgenArgs;


typedef struct {
	Token     *tokens;   
	Token     *tokens_end;
	Token     *tokens_first;   
	WrapgenArgs args;
} ParseCtx;

/*
	==========================================================
		Helper functions for parsing and handling errors
	==========================================================
*/

#define COUNT_ARRAY(x) ((int64_t)(sizeof(x)/sizeof(x[0])))

#define OPTIONAL(x) ((x), 1)
#define RESTORE(p)  (*p = p_saved);
#define SAVE(p) ParseCtx p_saved = *p; 
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

static int repr_type(int bufsz, char buf[], Type type) {

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


static int repr_symbol(int bufsz, char buf[], Symbol s) {

	long x = snprintf(buf, bufsz, "%s :=  ", s.name);
	bufsz -= x;
	buf += x;
	// TODO bug: make sure we haven't already overflown the buffer.

	return x + repr_type(bufsz, buf, s.type);
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
			fprintf(f, ">>HERE<< %s ", buf);
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
	
	static char *stringheap[1<<27] = {0};
	static struct ht stringtable = {
		.ht = stringheap,
		.exp = 27,
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


// TODO/NOTE not thread safe
// NOTE this is only a global symbol table
Symbol symtab[10000] = {0};
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

void clear_symbols(void)
{
	nsym = 0;
}

int modify_type_pointer(ParseCtx *p, Type *type)
{
	assert(type);
	if (type->is_pointer) 
		return 0;
	type->is_pointer = 1;
	return 1;
}

void modify_type_const(ParseCtx *p, Type *type)
{
	(void) p;
	assert(type);
	if(type->is_pointer) type->is_pointer_const = 1;
	else type->is_const = 1;
}

void modify_type_restrict(ParseCtx *p, Type *type)
{
	(void) p;
	assert(type);
	if(type->is_pointer) type->is_pointer_restrict = 1;
	else type->is_restrict = 1;
}

void modify_type_volatile(ParseCtx *p, Type *type)
{
	(void) p;
	assert(type);
	if(type->is_pointer) type->is_pointer_volatile = 1;
	else type->is_volatile = 1;
}

void modify_type_struct(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->category == T_UNINITIALIZED) type->category = T_STRUCT;
	else die(p, "'struct' does not make sense with '%s'", type_category_strings[type->category]);
}

void modify_type_union(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->category == T_UNINITIALIZED) type->category = T_UNION;
	else die(p, "'union' does not make sense with '%s'", type_category_strings[type->category]);
}

void modify_type_void(ParseCtx *p, Type *type)
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

void modify_type_char(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->is_imaginary)     die(p, "'char' does not make sense with '_Imaginary'");
	if(type->is_complex)       die(p, "'char' does not make sense with '_Complex'");
	if(type->category == T_UNINITIALIZED) type->category = T_CHAR;
	else die(p, "'char' does not make sense with '%s'", type_category_strings[type->category]);
}

void modify_type_short(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->is_imaginary)     die(p, "'short' does not make sense with '_Imaginary'");
	if(type->is_complex)       die(p, "'short' does not make sense with '_Complex'");
	if(type->category == T_UNINITIALIZED || type->category == T_INT) type->category = T_SHORT;
	else die(p, "'short' does not make sense with '%s'", type_category_strings[type->category]);
}

void modify_type_int(ParseCtx *p, Type *type)
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

void modify_type_long(ParseCtx *p, Type *type) 
{
	assert(type);
	if(type->category == T_UNKNOWN) return;

	if(type->category == T_UNINITIALIZED) type->category = T_LONG;
	else if (type->category == T_INT) type->category = T_LONG;
	else if (type->category == T_LONG) type->category = T_LLONG;
	else if (type->category == T_DOUBLE) type->category = T_LDOUBLE;
	else die(p, "'long' does not make sense with '%s'", type_category_strings[type->category]);
}

void modify_type_float(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->is_unsigned)     die(p, "'float' does not make sense with 'unsigned'");
	if(type->explicit_signed) die(p, "'float' does not make sense with 'signed'");

	if(type->category == T_UNINITIALIZED) type->category = T_FLOAT;
	else die(p, "'float' does not make sense with '%s'", type_category_strings[type->category]);
}

void modify_type_double(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->is_unsigned)     die(p, "'double' does not make sense with 'unsigned'");
	if(type->explicit_signed) die(p, "'double' does not make sense with 'signed'");

	if(type->category == T_UNINITIALIZED) type->category = T_DOUBLE;
	else if(type->category == T_LONG) type->category = T_LDOUBLE;
	else die(p, "'double' does not make sense with '%s'", type_category_strings[type->category]);
}

void modify_type_signed(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category >= T_FLOAT) die(p, "'signed' doesn't make sense with non-integer types");
	if(type->is_unsigned) die(p, "'signed' doesn't make sense with 'unsigned'");
	type->explicit_signed = 1;
}

void modify_type_unsigned(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category >= T_FLOAT) die(p, "'unsigned' doesn't make sense with non-integer types");
	if(type->is_unsigned) die(p, "'unsigned' doesn't make sense with 'signed'");
	type->is_unsigned = 1;
}

void modify_type_bool(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category == T_UNKNOWN) return;
	if(type->is_unsigned)     die(p, "'_Bool' does not make sense with 'unsigned'");
	if(type->explicit_signed) die(p, "'_Bool' does not make sense with 'signed'");

	if(type->category == T_UNINITIALIZED) type->category = T_BOOL;
	else die(p, "'_Bool' does not make sense with '%s'", type_category_strings[type->category]);
}

void modify_type_complex(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category >= T_CHAR && type->category < T_FLOAT) die(p, "'_Complex' doesn't make sense with integer types");
	if(type->is_imaginary) die(p, "'_Complex' doesn't make sense with '_Imaginary'");
	type->is_complex = 1;
}

void modify_type_imaginary(ParseCtx *p, Type *type)
{
	assert(type);
	if(type->category >= T_CHAR && type->category < T_FLOAT) die(p, "'_Imaginary' doesn't make sense with integer types");
	if(type->is_complex) die(p, "'_Imaginary' doesn't make sense with '_Complex'");
	type->is_imaginary = 1;
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

void emit_module(WrapgenArgs flags, int n_fnames, const char *fnames[])
{
	if(!flags.modulename) return;

	printf("static PyMethodDef module_functions[] = { \n");

	for (int i = 0; i < n_fnames; i++) {
		printf("{\"%s\", (PyCFunction) wrap_%s, METH_VARARGS|METH_KEYWORDS, \"\"},\n", 
			fnames[i], fnames[i]);
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

void emit_preamble(WrapgenArgs flags)
{
	(void) flags;
	static const char * preamble = 
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

int is_string(Type t)
{
	return t.category == T_CHAR
		&& !t.explicit_signed
		&& !t.is_unsigned
		&& t.is_pointer;
}

int is_voidptr(Type t)
{
	return t.category == T_VOID
		&& t.is_pointer;
}

int is_plainvoid(Type t)
{
	Type zero = {0};
	if(t.category == T_VOID){
		t.category = 0;
		return !memcmp(&t,&zero,sizeof(t));
	}
	return 0;
}

int is_array(Type t)
{
	return !is_string(t) 
		&& !is_voidptr(t)
		&& t.is_pointer;
}

Type basetype(Type t)
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

void emit_exceptionhandling(const char *fn, WrapgenArgs flags, int n_fnargs, Symbol fnargs[])
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

void emit_call(const char *fn, WrapgenArgs flags, int n_fnargs, Symbol fnargs[])
{
	printf("%s (", fn);

	for(int i = 0; i < n_fnargs; i++)
	{
		char *sep  =  i ? ", " : "";
		if (is_array(fnargs[i].type))
			printf("%sPyArray_DATA(%s)", sep, fnargs[i].name);
		else 
			printf("%s%s", sep, fnargs[i].name);
	}

	printf(");\n");
}

int emit_py_buildvalue_fmt_char(Type t) 
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
	else return 0;
	return 1;
}

void emit_wrapper (const char *fn, WrapgenArgs flags, int n_fnargs, Symbol fnargs[], Type rtntype)
{
	assert(n_fnargs >= 0);

	// declaration for function to be wrapped
	if(!flags.disable_declarations) {
		char buf[200] = {0};
		assert(sizeof(buf) > repr_type(sizeof(buf), buf, rtntype));

		printf("%s %s (", buf, fn);

		for(int i = 0; i < n_fnargs; i++)
		{
			memset(buf,0,sizeof(buf));
			assert(sizeof(buf) > repr_type(sizeof(buf), buf, fnargs[i].type));

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

			if (is_string(arg.type)) 
				printf("    const char * %s = 0;\n", arg.name);

			else if (is_voidptr(arg.type)) {
				printf("    unsigned long long %s_ull = 0;\n", arg.name);
				printf("    void * %s = 0;\n", arg.name);
			}

			else if (is_array(arg.type))
				printf("    PyArrayObject *%s = NULL;\n", arg.name);

			else {
				char buf[200] = {0};
				assert(sizeof(buf) > repr_type(sizeof(buf), buf, arg.type));
				printf("    %s %s = {0};\n", buf, arg.name);
			}
		}

		// parse python arguments into the above declared C variables
		printf("\n    if(!PyArg_ParseTupleAndKeywords(args, kwds, \"");
		for (int i = 0; i < n_fnargs; i++) {
			// building the format string for ParseTupleAndKeywords
			Symbol arg = fnargs[i];

			if      (is_string(arg.type))  printf("s");
			else if (is_voidptr(arg.type)) printf("K");
			else if (is_array(arg.type))   printf("O!");
			else {
				Type t = arg.type;
				if (!emit_py_buildvalue_fmt_char(t)) {

					char buf[200] = {0};
					assert(sizeof(buf) > repr_type(sizeof(buf), buf, t));
					die(0, "Error wrapping function '%s' in file '%s': "
					    "argument %i has type '%s', "
					    "which wrapgen doesn't know how to convert "
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
			else if (is_array(arg.type))   printf("&PyArray_Type, &%s", arg.name);
			else  printf("&%s", arg.name);

		}
		printf(")) return 0;\n\n");

		// type checking for any numpy arrays, conversions for any void pointers
		for (int i = 0; i < n_fnargs; i++)
		{
			Symbol arg = fnargs[i];

			if (is_voidptr(arg.type)) 
				printf("    memcpy(&%s, &%s_ull, sizoef(%s));\n", arg.name, arg.name, arg.name);

			else if (is_array(arg.type)) {
				char buf[200] = {0};
				assert(sizeof(buf) > repr_type(sizeof(buf), buf, basetype(arg.type)));

				// emit array type check
				printf("    if(PyArray_TYPE(%s) != C2NPY(%s)) {\n"
				       "        PyErr_SetString(PyExc_ValueError, \"Invalid array data type for argument '%s' (expected %s)\");\n"
			               "        return 0;\n"
				       "    }\n", arg.name, buf, arg.name, buf);

				// emit array contiguity check
				printf("    if(!PyArray_ISCARRAY(%s)) {\n"
				       "        PyErr_SetString(PyExc_ValueError, \"Argument '%s' is not C-contiguous\");\n"
			               "        return 0;\n"
				       "    }\n", arg.name, arg.name);
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
		assert(sizeof(buf) > repr_type(sizeof(buf), buf, rtntype));

		printf("    %s rtn = 0;\n", buf);
		printf("    Py_BEGIN_ALLOW_THREADS;\n");
		printf("    rtn = ");
		emit_call(fn, flags, n_fnargs, fnargs);
		printf("    Py_END_ALLOW_THREADS;\n");
		emit_exceptionhandling(fn, flags, n_fnargs, fnargs);
		printf("    return Py_BuildValue(\"s\", rtn);\n");

	} else if (is_voidptr(rtntype)) {

		char buf[200] = {0};
		assert(sizeof(buf) > repr_type(sizeof(buf), buf, rtntype));

		printf("    %s rtn = 0;\n", buf);
		printf("    Py_BEGIN_ALLOW_THREADS;\n");
		printf("    rtn = ");
		emit_call(fn, flags, n_fnargs, fnargs);
		printf("    Py_END_ALLOW_THREADS;\n");
		emit_exceptionhandling(fn, flags, n_fnargs, fnargs);
		printf("    return PyLong_FromVoidPtr(rtn);\n");

	} else if (is_array(rtntype)) {

		char buf[200] = {0};
		assert(sizeof(buf) > repr_type(sizeof(buf), buf, rtntype));

		die(0, "Error wrapping function '%s' in file '%s': "
		       "return type '%s' is not supported by wrapgen",
		       fn, flags.filename, buf);

	} else {
		char buf[200] = {0};
		assert(sizeof(buf) > repr_type(sizeof(buf), buf, rtntype));

		printf("    %s rtn = 0;\n", buf);
		printf("    Py_BEGIN_ALLOW_THREADS;\n");
		printf("    rtn = ");
		emit_call(fn, flags, n_fnargs, fnargs);
		printf("    Py_END_ALLOW_THREADS;\n");
		emit_exceptionhandling(fn, flags, n_fnargs, fnargs);
		printf("    Py_BuildValue(\"");
		if(!emit_py_buildvalue_fmt_char(rtntype)) {
			die(0, "Error wrapping function '%s' in file '%s': "
			       "return type '%s' is not supported by wrapgen",
			       fn, flags.filename, buf);
		}
		printf("\", rtn);\n");
	}

	printf("}\n\n");

}


/*
	==========================================================
		Parsing
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

static inline 
int check_token_is_identifier(Token *t, const char *id, long id_len)
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

int identifier(ParseCtx *p, char** out_id)
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

int typedef_name(ParseCtx *p, Type *t)
{
	if(p->tokens == p->tokens_end) return 0;
	if(p->tokens[0].toktype == CLEX_id) {

		Symbol *s = 0;
		if ((s = get_symbol(p->tokens[0].string))) {
			if(t) *t = s->type;
			p->tokens++;
			return 1;
		}
	}
	return 0;
}

int supported_type(ParseCtx *p, Type *t)
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
		if (eat_identifier(p, "volatile"))  { modify_type_restrict(p,t); return 1; }

		return 0;
	}
	return 0;	
}

int supported_type_list(ParseCtx *p, Type *t)
{
	if(!supported_type(p,t)) return 0;
	while(supported_type(p,t)) {}
	return 1;
}

int supported_typedef(ParseCtx *p, Symbol *s)
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

void populate_symbols(ParseCtx p)
{
	while(p.tokens != p.tokens_end) {

		while (!check_token_is_identifier(p.tokens, "typedef", 7)) 
		{ 
			p.tokens++;
			if (p.tokens == p.tokens_end) return;
		}

		Symbol s = {0};
		if (supported_typedef(&p, &s)) {
			add_symbol(s);

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


int arg(ParseCtx *p, const char *fn, Symbol *fnarg, int fatal)
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

int arglist(ParseCtx *p, const char *fn, int max_args, int *num_args, Symbol fnargs[], int fatal)
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

void attributes(ParseCtx *p, const char *fn)
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

void process_function(ParseCtx p)
{
	// on entry, p.tokens is set right on the function name.	
	const char * fn = p.tokens[0].string;
	// rewind to semicolon
	while(p.tokens[0].toktype != ';') p.tokens--;
	p.tokens++;

	Type rtn_t = {0};
	Symbol argsyms[40] = {0};

	if(!supported_type_list(&p, &rtn_t))
		die(&p, "error wrapping function '%s' in '%s': unsupported return type", fn, p.args.filename);
	
	char *sanity_check = 0;
	if(!identifier(&p, &sanity_check)) 
		die(&p, "error wrapping function '%s' in '%s': unsupported specifiers or qualifiers", fn, p.args.filename);
	if(sanity_check != fn)
		die(&p, "error wrapping function '%s' in '%s': parse error (encountered unrecognized garbage)", fn, p.args.filename);

	int num_args = 0;
	if(!arglist(&p, fn, 40, &num_args, argsyms, 0)) {
		arglist(&p, fn, 40, &num_args, argsyms, 1);
		die(&p, "error wrapping function '%s' in '%s': parse error (couldn't parse argument list)", fn, p.args.filename);
	}

	attributes(&p, fn);

	if(!eat_token(&p, ';'))
		die(&p, "error wrapping function '%s' in '%s': parse error (encountered unrecognized garbage)", fn, p.args.filename);



	// Parse successful; emit the wrapper!
	emit_wrapper (fn, p.args, num_args, argsyms, rtn_t);
}

void parse_file(ParseCtx p, const char *function_name)
{
	long len = strlen(function_name);

	while (p.tokens != p.tokens_end) {
		if (check_token_is_identifier(p.tokens, function_name, len) )
		{
			process_function(p);
			return;
		}
		p.tokens++;
	}
	die(0, "didn't find function called '%s' in file '%s'", function_name, p.args.filename);
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


void print_context(char *start, char *loc)
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
long delim_stack(enum delim_stack_action action, Token value, char *start, char* loc) {
	static short stack[200] = {0};
	static char *locations[200] = {0};
	static long pos = 0;

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

void delim_push(Token value, char *start, char* loc) { (void)delim_stack(DS_PUSH, value, start, loc); }
int  delim_pop(Token value, char *start, char* loc) { return delim_stack(DS_POP, value, start, loc); }
int  toplevel(void) { return delim_stack(DS_QUERY, (Token){0}, 0, 0); }


int lex_file(WrapgenArgs args, long long tokens_maxnum, Token *tokens, long long text_bufsz, char *text, long long string_store_bufsz, char *string_store)
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
			if(sizeof(path) <= snprintf(path, sizeof(path), "%s/%s", args.dirs[j], args.manual_include.files[i])) 
				die(0, "internal buffer overflow (path too long)");
			f = fopen(path, "rb");
			if(f) break;
		}

		// try the system default path
		if(!f) {
			char path[4096] = {0};
			// TODO accomodate windows
			if(sizeof(path) <= snprintf(path, sizeof(path), "/usr/include/%s", args.manual_include.files[i])) 
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
		char cmd[200] = {0};
		if(sizeof(cmd) <= snprintf(cmd, sizeof(cmd), "%s %s", args.preprocessor, args.filename)) 
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
				t.string = intern(lex.string, t.string_len);
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
		// when scanning braced code, check that delimiters mathc, but that's it. 

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


int usage(void)
{
	const char *message = 

	"wrapgen \n"
	"======= \n"
	"                                                                             \n"
	"Typical usage:  \n"
	" $ wrapgen -m coolmodule -f my_c_file.c function1 function2 > wrappers.c   \n"
	" $ cc -fPIC -shared -I/path/to/python/headers \\\n"
	"       wrappers.c my_c_file.c \\\n"
	"       -lpython -o coolmodule.so\n"
	"                                                                             \n"
	"The order of arguments matters, a flag affects everything after it, until a\n"
	"like flag overrides it's function. For example the command line  \n"
	"'-f file1.c sum product min max -f file2.c mean' \n"
	"would result in searching file1.c for functions called 'sum', 'product',  \n"
	"'min', 'max', and searching file2.c for a function called 'mean' \n"
	"                                                                             \n"
	"Flags: \n"
	"                                                                             \n"
	"-h   print help message and exit \n"
	"                                                                             \n"
	"-v   verbose \n"
	"                                                                             \n"
	"-D   disable including function declarations in the generated wrapper file \n"
	"     this might be used to facilitate amalgamation builds, for example \n"
	"                                                                             \n"
	"-m   the following argument is the name of the module to be built \n"
	"     only one module per wrapgen invocation is allowed  \n"
	"                                                                             \n"
	"-f   the following argument is a filename. \n"
	"                                                                             \n"
	"-p   the following argument specifies the preprocessor to use for the next \n"
	"     and future files, if different from the default 'cc -E' \n"
	"                                                                             \n"
	"-P   disable preprocessing of the next file encountered \n"
	"                                                                             \n"
	"-i   the following argument is a filename, to be inlcuded before the next  \n"
	"     file processed (for use with -P) \n"
	"                                                                             \n"
	"-I   the following argument is a directory path, to be searched for any  \n"
	"     future -i flags \n"
	"                                                                             \n"
	"-e   for functions that follow: if they return a string (const char *), the  \n"
	"     string is to be interpreted as an error message (if not null) and a python\n"
	"     exception should be thrown \n"
	"                                                                             \n"
	"     this flag only lasts until the next file change (i.e. -f) \n"
	"                                                                             \n"
	"-e,n,chkfn   for functions that follow: after calling, another function called\n"
	"     chkfn should be called.  chkfn should have the signature  \n"
	"     'const char * checkfn (?)' where ? is the type of the n-th argument to the\n"
	"     function (0 means the function's return value). if the chkfn call returns\n"
	"     a non-null string, that string is assumed to be an error message and a  \n"
	"     python exception is generated. \n"
	"                                                                             \n"
	"     this flag only lasts until the next file change (i.e. -f) \n";

	fprintf(stderr, "%s", message);
}

int 
main (int argc, char *argv[])
{
	(void)argc;
	argv++;

	if(!*argv) usage();

	/*
		Allocate buffers
	*/
	
	char  *text         = malloc(1<<27);
	char  *string_store = malloc(0x10000);
	Token *tokens       = malloc((1<<27) * sizeof(*tokens));
	if(!(text && string_store && tokens)) die(0,"out of mem");

	int emitted_preamble = 0;

	WrapgenArgs args = {.preprocessor = "cc -E"};
	int ntok = 0;

	const char *fnames[200] = {0};
	int n_fnames = 0;

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
				if(args.ndirs == COUNT_ARRAY(args.dirs)) die(0, "A maximum of %i -I flags can be used in a single wrapgen invocation.", (int)COUNT_ARRAY(args.dirs));
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

		if (!strcmp(*argv, "-f")) {
			argv++;

			if(!emitted_preamble) {
				emit_preamble(args); 
				emitted_preamble = 1;
			}

			if(*argv && **argv != '-') {
				memset(&args.error_handling, 0, sizeof(args.error_handling));
				args.filename = *argv;
				ntok = lex_file(args, 1<<27, tokens, 1<<27, text, 0x10000, string_store );
				clear_symbols();
				populate_symbols((ParseCtx) {
					.tokens_first  =  tokens,
					.tokens        =  tokens,
					.tokens_end    =  tokens+ntok,
					.args = args,
				});
				args.disable_pp = 0;
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
			die(0, "error: wrapgen only supports wrapping up to  %i functions", (int) COUNT_ARRAY(fnames));

		ParseCtx p = {
			.tokens_first  =  tokens,
			.tokens        =  tokens,
			.tokens_end    =  tokens+ntok,
			.args = args,
		};

		parse_file(p, *argv);
		argv++;
	}

	emit_module(args, n_fnames, fnames);
}
