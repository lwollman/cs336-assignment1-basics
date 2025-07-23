"""
Problem (unicode1): Understanding Unicode (1 point)
(a) What Unicode character does chr(0) return?
Deliverable: A one-sentence response.
(b) How does this character’s string representation (__repr__()) differ from its printed representa-
tion?
Deliverable: A one-sentence response.
(c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
Deliverable: A one-sentence response.
"""

"""
(a)
The hexadecimal representation of chr(0) is '\x00', which corresponds to the null character in Unicode.  We disabled code completion
to prevent VS code from autocompleting this.

(b)
This question doesn't seem totally well defined. WHen you say "this character" are you referring to '0' or chr(0)?

the repr of "0" is '0', i.e. the string "0" returns '0'

However, chr(0) 
chr(0).__repr__() is "'\\x00'"
as opposes to its string method
chr(0).__str__() is "'\x00'"


(c)

The print statement effectively ignores it, but forming a strig via the concatenate operator ("+") keeps the character around.

Bonus Point:
This is pretty intersting we tried a bunch of chr(x) where x is an integer and basically found that
the __str__, __repr__ methods were not always identical, and that the print statements behaviour 
when it encounters hexadecimal is not super easy to predict, \x00-\x07 all were ignored, but \x08 asked like a backspace? what! and then \x09 was like a tab.

print("qq" + chr(0) + "ww")
qqww
>>> print("qq" + chr(140000) + "ww")
qq𢋠ww



"""
