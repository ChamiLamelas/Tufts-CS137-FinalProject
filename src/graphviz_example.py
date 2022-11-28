# had to install this: https://graphviz.org/download/

import graphviz

dot = graphviz.Digraph('round-table', comment='The Round Table')  

print(dot.source)

dot.render(directory='test')