//
// Created by 廖田志浩 on 2024/7/14.
//

#ifndef QDLDL_SOLVER_QDLDL_INTERFACE_H
#define QDLDL_SOLVER_QDLDL_INTERFACE_H

#include <cstdlib>
#include <qdldl.h>
#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

inline vector<double> QDLDLSolve(const QDLDL_int &An,
                                 QDLDL_int *Ap,
                                 QDLDL_int *Ai,
                                 QDLDL_float *Ax,
                                 QDLDL_float *b) {
    QDLDL_int i; // Counter
    cout << "in qdldl solver\n";
    //data for L and D factors
    QDLDL_int Ln = An;
    QDLDL_int *Lp;
    QDLDL_int *Li;
    QDLDL_float *Lx;
    QDLDL_float *D;
    QDLDL_float *Dinv;

    //data for elim tree calculation
    QDLDL_int *etree;
    QDLDL_int *Lnz;
    QDLDL_int  sumLnz;

    //working data for factorisation
    QDLDL_int   *iwork;
    QDLDL_bool  *bwork;
    QDLDL_float  *fwork;

    //Data for results of A\b
    QDLDL_float *x;


    /*--------------------------------
     * pre-factorisation memory allocations
     *---------------------------------*/

    //These can happen *before* the etree is calculated
    //since the sizes are not sparsity pattern specific

    //For the elimination tree
    etree = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
    Lnz   = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);

    //For the L factors.   Li and Lx are sparsity dependent
    //so must be done after the etree is constructed
    Lp    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(An+1));
    D     = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
    Dinv  = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    //Working memory.  Note that both the etree and factor
    //calls requires a working vector of QDLDL_int, with
    //the factor function requiring 3*An elements and the
    //etree only An elements.   Just allocate the larger
    //amount here and use it in both places
    iwork = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(3*An));
    bwork = (QDLDL_bool*)malloc(sizeof(QDLDL_bool)*An);
    fwork = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    /*--------------------------------
     * elimination tree calculation
     *---------------------------------*/
    sumLnz = QDLDL_etree(An,Ap,Ai,iwork,Lnz,etree);

    /*--------------------------------
     * LDL factorisation
     *---------------------------------*/

    //First allocate memory for Li and Lx
    Li    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*sumLnz);
    Lx    = (QDLDL_float*)malloc(sizeof(QDLDL_float)*sumLnz);

    //now factor
    QDLDL_factor(An,Ap,Ai,Ax,Lp,Li,Lx,D,Dinv,Lnz,etree,bwork,iwork,fwork);

    /*--------------------------------
     * solve
     *---------------------------------*/
    x = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    //when solving A\b, start with x = b
    for(i=0;i < Ln; i++) x[i] = b[i];
    QDLDL_solve(Ln,Lp,Li,Lx,Dinv,x);

    vector<double> rst(x, x + An);

    /*--------------------------------
     * clean up
     *---------------------------------*/
    free(Lp);
    free(Li);
    free(Lx);
    free(D);
    free(Dinv);
    free(etree);
    free(Lnz);
    free(iwork);
    free(bwork);
    free(fwork);
    free(x);

    return rst;
}


#endif //QDLDL_SOLVER_QDLDL_INTERFACE_H
