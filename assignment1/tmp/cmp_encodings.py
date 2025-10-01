test_string ="hello! !"
encoded = test_string.encode("utf-32")

print(encoded)
print(type(encoded))
print(list(encoded))
print(f'test_string_len: {len(test_string)}')
print(f'encoded_len: {len(encoded)}')

# utf-8
# b'hello! !'
# <class 'bytes'>
# [104, 101, 108, 108, 111, 33, 32, 33]
# test_string_len: 8
# encoded_len: 8

# utf-16
# b'\xff\xfeh\x00e\x00l\x00l\x00o\x00!\x00 \x00!\x00'
# <class 'bytes'>
# [255, 254, 104, 0, 101, 0, 108, 0, 108, 0, 111, 0, 33, 0, 32, 0, 33, 0]
# test_string_len: 8
# encoded_len: 18

# utf-32
# b'\xff\xfe\x00\x00h\x00\x00\x00e\x00\x00\x00l\x00\x00\x00l\x00\x00\x00o\x00\x00\x00!\x00\x00\x00 \x00\x00\x00!\x00\x00\x00'
# <class 'bytes'>
# [255, 254, 0, 0, 104, 0, 0, 0, 101, 0, 0, 0, 108, 0, 0, 0, 108, 0, 0, 0, 111, 0, 0, 0, 33, 0, 0, 0, 32, 0, 0, 0, 33, 0, 0, 0]
# test_string_len: 8
# encoded_len: 36