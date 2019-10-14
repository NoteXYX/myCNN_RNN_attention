from numpy import concatenate, zeros

boc = 'abcdefghijklmnopqrstuvwxyz'

def getBOC():
	return boc

def char2idx( c ):
	v = (len(boc)+1)*[0]
	if c in boc:
		v[boc.index(c)] = 1
	else:
		v[len(boc)] = 1
	return v

def vector( word, length=30 ):
	wov = []
	if len(word) <= length:
		# wov = map(char2idx, word)			# old
		wov = list(map(char2idx, word))		# wov=[[0-26对应字母是否为1],[0-26对应字母是否为1]...]
		z =  zeros((length-len(wov), len(boc)+1))
		wov = concatenate((wov, z), axis=0)	# (30, 27)
		return wov
	else:
		return zeros((length, len(boc)+1))

