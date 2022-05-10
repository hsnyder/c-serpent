// WRAPGEN(mytestfn, tf2, tf3, tf4)

long mytestfn(int x, long y)
{
	return x + y;
}


void * tf2(void)
{
	return 0;
}


const char * tf3(int x)
{
	return "HALLOOOO";
}

const char * tf4(const char *mystrstr) {
	return mystrstr;
}
