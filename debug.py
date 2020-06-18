def print_dataset(dataset, logger):
    for i in range(len(dataset)):
        data = dataset[i]
        keys = ['old_set_indices', 'old_subgraph_indices', 'set_indices', 'edge_index', 'x', 'y']
        for key in keys:
            logger.info(key)
            logger.info(data.__dict__[key])
