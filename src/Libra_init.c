#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
Check these declarations against the C/Fortran source code.
*/

/* .C calls */
extern void ggm_C(void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void ising_C(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void LB_lasso(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void LB_logistic(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void LB_multi_logistic(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void potts_C(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);

static const R_CMethodDef CEntries[] = {
  {"ggm_C",               (DL_FUNC) &ggm_C,                9},
  {"ising_C",             (DL_FUNC) &ising_C,             11},
  {"LB_lasso",          (DL_FUNC) &LB_lasso,          15},
  {"LB_logistic",       (DL_FUNC) &LB_logistic,       15},
  {"LB_multi_logistic", (DL_FUNC) &LB_multi_logistic, 16},
  {"potts_C",             (DL_FUNC) &potts_C,             14},
  {NULL, NULL, 0}
};

void R_init_Libra(DllInfo *dll)
{
  R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
