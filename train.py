from sklearn.metrics import roc_auc_score
from utils import *
criterion = torch.nn.functional.cross_entropy


def train_model(model, dataloaders, args, logger):
    device = get_device(args)
    model.to(device)
    train_loader, val_loader, test_loader = dataloaders
    optimizer = get_optimizer(model, args)
    metric = args.metric
    recorder = Recorder(metric)
    for step in range(args.epoch):
        optimize_model(model, train_loader, optimizer, device)
        train_loss, train_acc, train_auc = eval_model(model, train_loader, device)
        val_loss, val_acc, val_auc = eval_model(model, val_loader, device)
        test_loss, test_acc, test_auc = eval_model(model, test_loader, device)
        recorder.update(train_acc, train_auc, val_acc, val_auc, test_acc, test_auc)

        logger.info('epoch %d best test %s: %.4f, train loss: %.4f; train %s: %.4f val %s: %.4f test %s: %.4f' %
                    (step, metric, recorder.get_best_metric(val=True)[0], train_loss,
                     metric, recorder.get_latest_metrics()[0], metric, recorder.get_latest_metrics()[1],
                     metric, recorder.get_latest_metrics()[2]))
    logger.info('(With validation) final test %s: %.4f (epoch: %d, val %s: %.4f)' %
                (metric, recorder.get_best_metric(val=True)[0],
                 recorder.get_best_metric(val=True)[1], metric, recorder.get_best_val_metric(val=True)[0]))
    logger.info('(No validation) best test acc: %.4f (epoch: %d)' % recorder.get_best_acc(val=False))
    logger.info('(No validation) best test auc: %.4f (epoch: %d)' % recorder.get_best_auc(val=False))

    return recorder.get_best_metric(val=True)[0], recorder.get_best_metric(val=False)[0]


def optimize_model(model, dataloader, optimizer, device):
    model.train()
    # setting of data shuffling move to dataloader creation
    for batch in dataloader:
        batch = batch.to(device)
        label = batch.y
        prediction = model(batch)
        loss = criterion(prediction, label, reduction='mean')
        loss.backward()
        optimizer.step()


def eval_model(model, dataloader, device, return_predictions=False):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            labels.append(batch.y)
            prediction = model(batch)
            predictions.append(prediction)
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
    if not return_predictions:
        loss, acc, auc = compute_metric(predictions, labels)
        return loss, acc, auc
    else:
        return predictions


def compute_metric(predictions, labels):
    with torch.no_grad():
        # compute loss:
        loss = criterion(predictions, labels, reduction='mean').item()
        # compute acc:
        correct_predictions = (torch.argmax(predictions, dim=1) == labels)
        acc = correct_predictions.sum().cpu().item()/labels.shape[0]
        # compute auc:
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        multi_class = 'ovr'
        if predictions.size(1) == 2:
            predictions = predictions[:, 1]
            multi_class = 'raise'
        auc = roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy(), multi_class=multi_class)
    return loss, acc, auc


class Recorder:
    """
    always return test numbers except the last method
    """
    def __init__(self, metric):
        self.metric = metric
        self.train_accs, self.val_accs, self.test_accs, self.train_aucs, self.val_aucs, self.test_aucs = [], [], [], [], [], []

    def update(self, train_acc, train_auc, val_acc, val_auc, test_acc, test_auc):
        self.train_accs.append(train_acc)
        self.train_aucs.append(train_auc)
        self.val_accs.append(val_acc)
        self.test_accs.append(test_acc)
        self.val_aucs.append(val_auc)
        self.test_aucs.append(test_auc)

    def get_best_metric(self, val):
        dic = {'acc': self.get_best_acc(val=val), 'auc': self.get_best_auc(val=val)}
        return dic[self.metric]

    def get_best_acc(self, val):
        if val:
            max_step = int(np.argmax(np.array(self.val_accs)))
        else:
            max_step = int(np.argmax(np.array(self.test_accs)))
        return self.test_accs[max_step], max_step

    def get_best_auc(self, val):
        if val:
            max_step = int(np.argmax(np.array(self.val_aucs)))
        else:
            max_step = int(np.argmax(np.array(self.test_aucs)))
        return self.test_aucs[max_step], max_step

    def get_latest_metrics(self):
        if len(self.train_accs) < 0:
            raise Exception
        if self.metric == 'acc':
            return self.train_accs[-1], self.val_accs[-1], self.test_accs[-1]
        elif self.metric == 'auc':
            return self.train_aucs[-1], self.val_aucs[-1], self.test_aucs[-1]
        else:
            raise NotImplementedError

    def get_best_val_metric(self, val):
        max_step = self.get_best_auc(val=val)[1]
        dic = {'acc': (self.val_accs[max_step], max_step), 'auc': (self.val_aucs[max_step], max_step)}
        return dic[self.metric]
