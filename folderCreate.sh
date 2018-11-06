#!/bin/bash
cd ./branco
num=1
for file in $(ls -v *.jpg); do
       mkdir "/home/giuliano/Documentos/UFES_2018_1/P.G/branco_lesoes/"$num
       let num=$num+1
done
