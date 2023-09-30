#include<stdio.h>
#include<string.h>
#include<stdlib.h>

//versao 1 compara 2 outputs do ficheiro ompMat e do seqMat
void compareFiles(FILE *file1, FILE *file2){
    char ch1 = getc(file1);
    char ch2 = getc(file2);
    int error = 0, pos = 0, line = 1;
    while (ch1 != EOF && ch2 != EOF){
        pos++;
        if (ch1 == '\n' && ch2 == '\n'){
            line++;
            pos = 0;
        }
        if (ch1 != ch2){
            error++;
            printf("Line Number : %d \tError"
                   " Position : %d \n", line, pos);
        }
        ch1 = getc(file1);
        ch2 = getc(file2);
    }
    printf("Total Errors : %d\t \n", error);
}
int main(int argc, char *argv[]){

    FILE *file1;
    FILE *file2;

    for (size_t i = 0; i <=100000; i += 10000)
    {
        char filename1[64];
        char filename2[64];
        /* code */
        sprintf(filename1, "heat_%06d.pgm", n);
        sprintf(filename2, "P_heat_%06d.pgm", n);
    
    
        file1 = fopen(filename1, "r");
        file2 = fopen(filename2, "r");
    

    if (file1 == NULL || file2 == NULL){
        printf("Error : Files not open");
        exit(0);
    }
    compareFiles(file1, file2);
    fclose(file1);
    fclose(file2);}
    return 0;
}