# python 2.7

import re

str = 'an example word:cat'

match = re.findall(r'...', str)

print match
#
# if match:
#     print 'found', match.group()
# else:
#     print 'found nothing'

match = re.search(r'^[\w\s]+$', 'p iii ggg')
print match.group()
