import glob, zlib, base64, hashlib
B = glob.glob('/kaggle/input/**/task2_bundle', recursive=True)[0]
b = open(B + '/t2lib.py', 'rb').read()
print('SHA', hashlib.sha256(b).hexdigest(), len(b))
z = base64.b64encode(zlib.compress(b, 9)).decode()
print('LEN', len(z))
for i in range(0, len(z), 400):
    print('Z%03d' % (i // 400), z[i:i + 400])
print('END')
