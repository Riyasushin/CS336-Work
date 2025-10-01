# Problem （unicode1）: Understanding Unicode （1 point）

```python
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
>>>
```

# Problem(unicode2): Unicode Encodings(3 points)

## (a)

- UTF-8 为变长编码，对 ASCII 字符（文本中常见）仅用 1 字节，存储和处理更高效；
- UTF-8 兼容 ASCII，在多语言文本中适应性更强
- 而 UTF-16/32 对 ASCII 文本冗余度高（UTF-16 至少 2 字节，UTF-32 固定 4 字节）。

## (b)

输入 日语或中文即可

## (c)

指的是不符合 UTF-8 规则的
