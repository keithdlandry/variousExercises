from __future__ import division

def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


n = 0
for x in xrange(1,300):
	for y in xrange(1,300):

		#n = n + pow(2/3, 2*x-1)*pow(1/3, y)
		arr = choose(x+y-2, y-1)
		#print 'i = ', x, 'j = ', y, '# arrange = ', arr
		n = n + arr*pow(2/3, 2*x-1)*pow(1/3, y)*(2*x + 2*y -1)

print 'expected number of transitions = ', n
		