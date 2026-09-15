import copy
import logging
import torch
from torch.utils.data import DataLoader

from client import Client
from server import Server
from utils import setup_logging, distribute_data_dirichlet, distribute_data, fix_random_seed
from args import get_parse_args
from prepare_datasets import get_datasets
from get_model import get_model

import warnings
warnings.filterwarnings("ignore")


def start_training(args):
    
    # Get datasets and partition data among clients
    train_dataset, test_dataset, num_target, args.training_rounds = get_datasets(args.dataset, args.data_path)
    assert args.num_clients <= num_target, "The number of clients should be less than or equal to the number of classes."

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    if args.non_iid:
        user_groups = distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = distribute_data(train_dataset, args, n_classes=num_target)

    # Create clients, assign data to each client, and record the data size of each client for weighted aggregation
    clients, client_data_sizes = [], {}
    for _id in range(0, args.num_clients):
        client = Client(_id, args, train_dataset, user_groups[_id])
        client_data_sizes[_id] = client.n_data
        clients.append(client)

    # Initialize the server with the data sizes of clients and the test loader for the main task
    server = Server(client_data_sizes, args, test_loader)
    
    # Intialize the model and copy it for each client
    initial_model = get_model(args.dataset).to(args.device)
    
    ########
    # Please note that this process may consume a lot of GPU memory according to the number of clients and the model size.
    # If you encounter out-of-memory error, please consider reducing the number of clients or using a smaller model.
    ########
    client_model_dict = {}
    for i in range(args.num_clients):
        client_model_dict[i] = copy.deepcopy(initial_model)

    ####################
    ## Start training ##
    ####################
    
    for rnd in range(1, args.training_rounds + 1):
        logging.info("--------round {} ------------".format(rnd))

        # By default we do not perform client sampling, but you can easily modify the code to do so.
        chosen = [i for i in range(args.num_clients)]

        for client_id in chosen:
            clients[client_id].local_train(client_model_dict[client_id], rnd)

        # Aggregate the updated models from all clients
        client_model_dict = server.aggregation(client_model_dict, rnd)

    logging.info('Best results:')
    logging.info('ACC:              %.2f' % server.best_acc)
    # Please note that the following best VR may not be the VR of the model with the best ACC, as we do not select the best model based on VR.
    logging.info('Veri:             %.2f' % server.best_veri)
    logging.info('Training has finished!')


if __name__ == "__main__":

    args = get_parse_args()
    
    fix_random_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.log_dir = setup_logging(args)
    
    start_training(args)

