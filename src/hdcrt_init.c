#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .Call calls */
extern SEXP _SPR_Test(void *, void *, void *, void *, void *);
extern SEXP _SPR_Test_Dependent_Censor(void *, void *, void *, void *, void *);
extern SEXP _SPR_Test_Double_Censor(void *, void *, void *, void *, void *, void *);
extern SEXP _SPR_Test_Part(void *, void *, void *, void *, void *, void *);
extern SEXP _SPR_Test_Part_v2(void *, void *, void *, void *, void *, void *);
extern SEXP _SPRFABS(void *, void *, void *, void *, void *, void *, void *, void *, void *);

static const R_CallMethodDef CallEntries[] = {
    {"_SPR_Test",                  (DL_FUNC) &_SPR_Test,                  5},
    {"_SPR_Test_Dependent_Censor", (DL_FUNC) &_SPR_Test_Dependent_Censor, 5},
    {"_SPR_Test_Double_Censor",    (DL_FUNC) &_SPR_Test_Double_Censor,    6},
    {"_SPR_Test_Part",             (DL_FUNC) &_SPR_Test_Part,             6},
    {"_SPR_Test_Part_v2",          (DL_FUNC) &_SPR_Test_Part_v2,          6},
    {"_SPRFABS",                   (DL_FUNC) &_SPRFABS,                   9},
    {NULL, NULL, 0}
};

void R_init_hdcrt(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}