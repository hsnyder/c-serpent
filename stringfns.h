/* 
	This "library" is "STB style" (see https://github.com/nothings/stb)
	Short version: the header includes both the declarations and the implementation.	
	The implementation is compiled out unless STRINGFNS_IMPLEMENTATION is defined.
	So to use this library, you need to define STRINGFNS_IMPLEMENTATION in ONE .c file
	before you include this header. 
*/

#ifndef STRINGFNS_H
#define STRINGFNS_H
#include <wchar.h>

/*
	Trim leading and trailing whitespace from (wide) character string, in-place.
*/

void
wstr_trim_inplace(wchar_t * str);

void
str_trim_inplace(char * str);

char *
skipnl (char *t);

char *
skipst (char *t);

char * 
skipwhitespace(char *t);

char * 
strsepstr(char **string, const char *delim);

int
remove_substr(char * str, const char * sub);

#endif

#ifdef STRINGFNS_IMPLEMENTATION

#include <wctype.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

void
wstr_trim_inplace(wchar_t * str)
{
	/*
		Trim leading and trailing whitespace from wide character string, in-place.
	*/
	const size_t len = wcslen(str);
	if(len == 0) return;
	size_t first = 0;
	while(first < len && iswspace(str[first])) first++;
	size_t last = len-1;
	while(last > first && iswspace(str[last])) last--;
	wmemmove(str,&str[first],last-first+1);
	str[last-first+1] = 0;
}

void
str_trim_inplace(char * str)
{
	/*
		Trim leading and trailing whitespace from string, in-place.
	*/
	const size_t len = strlen(str);
	if(len == 0) return;
	size_t first = 0;
	while(first < len && isspace(str[first])) first++;
	size_t last = len-1;
	while(last > first && isspace(str[last])) last--;
	memmove(str,&str[first],last-first+1);
	str[last-first+1] = 0;
}

char *
skipnl (char *t)
{
        while(*t=='\r' || *t=='\n') t++;
        return t;
}

char *
skipst (char *t)
{
        while (*t=='\r' || *t==' ') t++;
        return t;
}

char * 
skipwhitespace(char *t) 
{
        while (isspace(*t)) t++;
        return t;
}

char * 
strsepstr(char **string, const char *delim)
{
	/*
		1. Finds the next occurrence of delim in *string.
		2. Replaces said occurrence with 0s
		3. Upates *string to point just past the found delim.
		4. Returns the original value of *string.

		If no occurrences of delim are found, *string will point to the null terminating char after return
	*/

	const size_t dlen = strlen(delim);
	char * retval = *string;

	char * x = strstr(*string, delim);

	if(x) {
		// x points to the first occurrence of delim
		memset(x, 0, dlen);
		*string = x + dlen;
	} else {
		// no occurrences of delim
		*string += strlen(*string);
	}
	return retval;
}

int 
remove_substr(char * str, const char * sub) 
{
	char *x = str;
	size_t sublen = strlen(sub);
	int r = 0;
	while((x = strstr(x, sub))) {
		memset(x,' ',sublen);
		x += sublen;
		r++;
	}
	str_trim_inplace(str);
	return r;
}

#endif
#ifdef STRINGFNS_SELF_TEST

int main(void)
{
	{
		wchar_t test[100] = L" Hello   \n\r  ";
		wprintf(L"Before: '%ls'\n", test);
		wstr_trim_inplace(test);
		wprintf(L"After:  '%ls'\n", test);
	}
	{
		wchar_t test[100] = L"Hello   \n\r  ";
		wprintf(L"Before: '%ls'\n", test);
		wstr_trim_inplace(test);
		wprintf(L"After:  '%ls'\n", test);
	}
	{
		wchar_t test[100] = L" Hello";
		wprintf(L"Before: '%ls'\n", test);
		wstr_trim_inplace(test);
		wprintf(L"After:  '%ls'\n", test);
	}
	{
		wchar_t test[100] = L"Hello";
		wprintf(L"Before: '%ls'\n", test);
		wstr_trim_inplace(test);
		wprintf(L"After:  '%ls'\n", test);
	}


	{
		char test[100] = " Hello   \n\r  ";
		printf("Before: '%s'\n", test);
		str_trim_inplace(test);
		printf("After:  '%s'\n", test);
	}
	{
		char test[100] = "Hello   \n\r  ";
		printf("Before: '%s'\n", test);
		str_trim_inplace(test);
		printf("After:  '%s'\n", test);
	}
	{
		char test[100] = " Hello";
		printf("Before: '%s'\n", test);
		str_trim_inplace(test);
		printf("After:  '%s'\n", test);
	}
	{
		char test[100] = "Hello";
		printf("Before: '%s'\n", test);
		str_trim_inplace(test);
		printf("After:  '%s'\n", test);
	}
}

#endif
