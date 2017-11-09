from utils import Emailer

em = Emailer('results74207281@gmail.com', 'deeplearning','psn240@nyu.edu')

em.send_msg('test', 'test', ['sample/rocs_conv.png'])
