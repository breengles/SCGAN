#!/usr/bin/env bash


for cfg in configs/overfit/*
do
  ./train.py "$cfg" --local
done
