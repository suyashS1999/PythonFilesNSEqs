#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:28:59 2018

@author: Misa
"""

import time
from math import floor

stepabcd = 1
countabcd = 0
starttimex = 0
globalrvarmytools = 0

def header_time_s(maxv,name):
	starttimex = time.time()
	x = 0
	if maxv - 1 > 50:
		strlen = 50
		stepabcd = (maxv - 1)/50
	else:
		strlen = maxv-1
		stepabcd = 1
	app = (strlen-len(name))/2
	if app%1 != 0:
		x = 1
	if app > 0:
		app = floor(app)
		str1 = app*"-"+name+app*"-"+x*"-"
	else:
		str1 = name
	return starttimex,stepabcd,str1

def pbf(variable, maxv, name = "Progress"):
	if variable == 0:
		global starttimex,stepabcd,countabcd
		starttimex,stepabcd,str1 = header_time_s(maxv,name)
		print(str1)
	cals = variable/stepabcd
	n = floor(cals - countabcd)
	if n == 1:
		countabcd = int(cals)
		print("*", sep='', end='', flush=True)
	if variable == maxv-1:
		e = time.time()
		print("     ==",round((e-starttimex),4),"seconds ==")
		stepabcd = 1
		countabcd = 0
		starttimex = 0
		
def pbw(variable, maxv,dv = 0, name = "Progress"):
	if variable == 0:
		global starttimex,stepabcd,countabcd
		starttimex,stepabcd,str1 = header_time_s(51,name)
		stepabcd = maxv/51
		print(str1)
	cals = variable/stepabcd
	n = floor(cals - countabcd)
	if n == 1:
		countabcd = int(cals)
		print("*", sep='', end='', flush=True)
	if round(variable,4) == maxv-dv:
		e = time.time()
		print("     == ",round((e-starttimex),4),"seconds ==")
		stepabcd = 1
		countabcd = 0
		starttimex = 0
			 
		
		
def timeit(variable,maxv,dv = 1,split_time = 0,name = "Time elapsed"):
	global starttimex
	if variable == 0:
		starttimex = time.time()
	if variable == maxv-dv:
		e = time.time()
		print(name,"  == ",round((e-starttimex),4),"seconds ==")
		if split_time == 1:
			x = (e-starttimex)/(maxv/dv)
			n = "second"
			if x < 0.001:
				x *= 1000
				n = "ms"
				if x < 0.001:
					x *= 1000
					n = "µs"
			print(round(x,4)," ",n," per loop")
		starttimex = 0