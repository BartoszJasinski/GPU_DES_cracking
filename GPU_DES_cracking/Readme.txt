Keys in my program are 14 digits long (in hex), you can check if DES output matches output of this site: http://des.online-domain-tools.com/ 
if you will perform following steps:
In DES every 8th bit is not taken to computation i.e. 8, 16, 24, 32, 40, 48, 56, 64. So I use 56bit long key, and implementation 
on above site uses 64bit long key(which is still 56bit long effectivly). So that the outputs of my program and site agree binary representation
of keys after deletation of every 8th bit should be the same e.g.
Key -> site key
00000000000000 -> 0000000000000000
0B000000000000 -> 0B80000000000000
70000000000000 -> 7000000000000000
000000AC000000 -> 00000014C0000000
and so on.

