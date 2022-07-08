// WRAPGEN mytestfn tf2    tf3

long mytestfn(int x, long y)
{
	return x + y;
}


void * tf2(void)
{
	return 0;
}


void asdf(void * x) {
	int * y = x;
	*y = 0;
}


const char * tf3(int x)
{
	return "HALLOOOO";
}

const char * tf4(const char *mystrstr) {
	return mystrstr;
}

/*
 *
 * WRAPGEN -e1 asdf tf4
 */
