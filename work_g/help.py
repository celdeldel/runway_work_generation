#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: celdel

created to have a nice and clean encods_1.csv with all encods of all the img
"""

import tsv 
import pandas as pd
import csv



reader = tsv.TsvReader(open("encods.tsv"))
new_list = []
for parts in reader :
    new_list.append(list(parts))

df = pd.DataFrame(new_list)
df.to_csv("encods_1.csv")
