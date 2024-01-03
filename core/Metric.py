import os
import csv
import torch
import numpy as np
'''
to do : create new func to deal with data mining task(select bad dice per subjects).
'''
class Metric:
    '''Metric computes accuracy/precision/recall/confusion_matrix with batch updates.'''

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.label_name = [f'dice_{str(i)}' for i in range(self.num_classes)]
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.tn = torch.zeros(self.num_classes)

    def update(self, y, t):
        '''Update with batch outputs and labels.
        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].
        '''
        y, t = self._process_pixel(y, t)
        self._process(y, t)


    def _process_pixel(self, y, t):
        y = torch.argmax(y, 1)
        t = torch.argmax(t, 1)
        return y, t

    def _process(self, y, t):
        '''Compute TP, FP, FN, TN.
        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].
        Returns:
          (tensor): TP, FP, FN, TN, sized [num_classes,].
        '''
        for i in range(self.num_classes):
            self.tp[i] += ((y == i) & (t == i)).sum().item()
            self.fp[i] += ((y == i) & (t != i)).sum().item()
            self.fn[i] += ((y != i) & (t == i)).sum().item()
            self.tn[i] += ((y != i) & (t != i)).sum().item()

    def dice(self):
        dice = torch.empty(self.num_classes)
        for c in range(self.num_classes):
            dice[c] = 2 * self.tp[c] / (2 * self.tp[c] + self.fp[c] + self.fn[c])
        return {self.label_name[i]: format(dice[i].item(), '.6f') for i in range(self.num_classes)}

    def iou(self):
        iou = torch.empty(self.num_classes)
        for c in range(self.num_classes):
            iou[c] = self.tp[c] / (self.tp[c] + self.fp[c] + self.fn[c])
        return {self.label_name[i]: format(iou[i].item(), '.6f') for i in range(self.num_classes)}

    def accuracy(self, reduction='mean'):
        '''Accuracy = (TP+TN) / (P+N).
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) accuracy.
        '''
        assert(reduction in ['none', 'mean'])
        # tp = torch.cat(self.tp, 0)
        # fn = torch.cat(self.fn, 0)
        if reduction == 'none':
            acc = self.tp / (self.tp + self.fn)
        else:
            acc = self.tp.sum() / (self.tp + self.fn).sum()
        return acc

    def __precision(self, reduction='mean'):
        '''Precision = TP / (TP+FP).
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) precision.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        tp, fp, fn, tn = self._process(y, t)
        prec = tp / (tp + fp)
        prec[torch.isnan(prec)] = 0
        if reduction == 'mean':
            prec = prec.mean()
        return prec

    def __recall(self, reduction='mean'):
        '''Recall = TP / P.
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) recall.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        tp, fp, fn, tn = self._process(y, t)
        recall = tp / (tp + fn)
        recall[torch.isnan(recall)] = 0
        if reduction == 'mean':
            recall = recall.mean()
        return recall

    def __confusion_matrix(self):
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        matrix = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                matrix[j][i] = ((y == i) & (t == j)).sum().item()
        return matrix

    def __f1(self, each_cls=False):
        if each_cls:
            prec = self.precision(reduction='none')
            recall = self.recall(reduction='none')

        else:
            prec = self.precision()
            recall = self.recall()
        return 2 * prec * recall / (prec + recall)


def test():
    import pytorch_lightning.metrics.functional as M
    nc = 10
    m = Metric(num_classes=nc)
    y = torch.randint(0, nc, (10,))
    t = torch.randint(0, nc, (10,))
    m.update(y, t)

    print('\naccuracy:')
    print(m.accuracy('none'))
    print(M.accuracy(y, t, nc, 'none'))
    print(m.accuracy('mean'))
    print(M.accuracy(y, t, nc, 'elementwise_mean'))

    print('\nprecision:')
    print(m.precision('none'))
    print(M.precision(y, t, nc, 'none'))
    print(m.precision('mean'))
    print(M.precision(y, t, nc, 'elementwise_mean'))

    print('\nrecall:')
    print(m.recall('none'))
    print(M.recall(y, t, nc, 'none'))
    print(m.recall('mean'))
    print(M.recall(y, t, nc, 'elementwise_mean'))

    print('\nconfusion matrix:')
    print(m.confusion_matrix())
    print(M.confusion_matrix(y, t))


if __name__ == '__main__':
    test()