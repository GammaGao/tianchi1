#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

import ConfigParser


class train_config():
    def __init__(self, filename='train.ini'):
        conf = ConfigParser.ConfigParser()
        conf.read(filename)
        self.train = conf.get('train', 'train')
        self.testa = conf.get('train', 'testa')
        self.testb = conf.get('train', 'testb')

    def get_train_settings(self):
        return (self.train, self.testa, self.testb)
