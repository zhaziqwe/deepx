
#include <iostream>
#include "deepx/tf/arg.hpp"
#include "deepx/tf/tf.hpp"
#include "deepx/mem/mem.hpp"
using namespace std;
using namespace deepx::tf;
 
int main(int argc,char **argv){
    ArgSet argsetdef;
    argsetdef.funcdef();
    cout<<argsetdef.to_string()<<endl;
    ArgSet argset("argset(1.08)->(var<float32> a)",true);
    cout<<argset.to_string()<<endl;
    Mem mem;
    string error;
    int code=argset.run(mem,error);
    if(code!=0){
        cout<<error<<endl;
    }
    cout<<mem.getarg<float>("a")<<endl;
    return 0;
}