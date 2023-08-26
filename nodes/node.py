
from NodeGraphQt import BaseNode

from typing import Tuple
from numbers import Number

class Node(BaseNode):
    """
    """
    # unique node identifier.
    __identifier__ = 'nodes.node'

    # initial default node name.
    NODE_NAME = 'Node'

    def __init__(self):
        super().__init__()

        # Set up the initial attributes
        self.panel = None

        # create node outputs.
        self.add_output('out')

    def get_input_node(self, index: int):
        input_port = self.input(index)
        connected_ports = input_port.connected_ports()
        predecessor_node = connected_ports[0].node() if connected_ports else None

        return predecessor_node

    def frame_range(self)-> Tuple[Number, Number]: 
        predecessor_node = self.get_input_node(0)

        if not predecessor_node:
            return None, None

        return predecessor_node.frame_range()

    def get_image_data(self, frame):
        return None


# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     # theme.set_theme(app, theme='dark')
#     panel = NodePanel()
#     panel.show()
#     sys.exit(app.exec_())
