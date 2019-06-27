#!/bin/bash

for filename in `pwd`/*.off; do
    fn=$(basename "$filename" .off)
    printf $fn
    printf "\n"
    meshlabserver -i $fn".off" -o "obj/"$fn".obj"
done
