def get_framework_parsers(parser):
    
    parser.add_argument('--dataset',type=str, default='chameleon', help='The name of datasets')
    parser.add_argument('--model',type=str, default='gprfpa', help='The name of models')#gprfpa
    parser.add_argument('--runs',type=int, default=10, help='The num of experiments')
    parser.add_argument('--epochs',type=int, default=1000, help='The number of iterations')
    parser.add_argument('--train_rate', type=float, default=0.6, help='The percentage of the training set')
    parser.add_argument('--val_rate', type=float, default=0.2, help='The percentage of the validation set')
    parser.add_argument('--lr', type=float, default=0.01,help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='The decay rate of weights')
    parser.add_argument('--hidden_channels', type=int, default=64, help='The dimension of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate')
    parser.add_argument('--K', type=int, default=10, help='The number of layers of baseline models')
    parser.add_argument('--alpha', type=float, default=0.2, help='The hyperparameter of APPNP')
    parser.add_argument('--eta',type=float,default=0., help='The hypterparameter of (S)FPA-master')
    parser.add_argument('--early_stopping', type=int, default=100)
    parser.add_argument('--simp', type=bool, default=False, help='(Simplified)FPA-master') #if it is true, y_onehot element * aug; else y_onehot - aug
