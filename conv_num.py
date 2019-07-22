#!/usr/bin/env python3
  
username = ['inusan', 'kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan', 'kubosan', 'sarada']
#old username = ['inusan', 'kubosan', 'kumasan', 'nekosan', 'sarada', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']
  

def conv_num(name):
    return username.index(name)+1

def conv_list(name_list):
    num_list = []
    for name in name_list:
        num_list.append(conv_num(name))
        
    return num_list

if __name__ == '__main__':

    username2 = ['inusan', 'kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']
    print(conv_list(username2))

