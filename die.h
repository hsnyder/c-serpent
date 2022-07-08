#ifndef _DIE_H
#define _DIE_H
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <stdarg.h>
#include <string.h>

#ifndef DIE_H_OUTPUT_MESSAGE
#define DIE_H_OUTPUT_MESSAGE(m) fputs(m,stderr)
#endif

static _Noreturn void 
#if defined(__clang__) || defined(__GNUC__)
__attribute__ ((format (printf, 1, 2)))
#endif
die(char *fmt, ...)
{
	char buf[1024]  = {};
	char buf2[128]  = {};
	char buf3[1024] = {};

	int e = errno;
	if (e != 0) snprintf(buf2,sizeof(buf2)," (errno %d: %s)", e, strerror(e));

	va_list args;
	va_start(args, fmt);
	vsnprintf(buf, sizeof(buf), fmt, args);
	va_end(args);

	snprintf(buf3, sizeof(buf3), "%s%s\n", buf, buf2);
	DIE_H_OUTPUT_MESSAGE(buf3);
#ifdef DIE_ABORT
	abort();
#else 
	exit(EXIT_FAILURE);
#endif
}

static void 
#if defined(__clang__) || defined(__GNUC__)
__attribute__ ((format (printf, 1, 2)))
#endif
warn(char *fmt, ...)
{
	char buf[1024]  = {};
	char buf2[128]  = {};
	char buf3[1024] = {};

	int e = errno;
	if (e != 0) snprintf(buf2,sizeof(buf2)," (errno %d: %s)", e, strerror(e));

	va_list args;
	va_start(args, fmt);
	vsnprintf(buf, sizeof(buf), fmt, args);
	va_end(args);

	snprintf(buf3, sizeof(buf3), "%s%s\n", buf, buf2);
	DIE_H_OUTPUT_MESSAGE(buf3);
}

#ifdef NDEBUG
#define xassert(cond) do{(void)(cond)}while(0);
#else
#define xassert(cond) if(!(cond)){ die("Assertion failed %s:%i %s", __FILE__, __LINE__, #cond); }
#endif

static inline char *
#if defined(__clang__) || defined (__GNUC__)
__attribute__ ((format (printf, 3, 4)))
#endif
fmt_str (int n, char *restrict buffer, const char *restrict fmt, ...)
{
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buffer, n, fmt, ap);
        va_end(ap);
        return buffer;
}

#define NFORMAT(N, fmt, ...) fmt_str(N, (char[N]){0}, (fmt), __VA_ARGS__)

#endif
