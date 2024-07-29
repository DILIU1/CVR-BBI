// This file has been generated by Py++.

#include "boost/python.hpp"
#include "wrappable_v3d.h"
#include "RGB32f.pypp.hpp"

namespace bp = boost::python;

void register_RGB32f_class(){

    bp::class_< RGB32f >( "RGB32f" )    
        .def_readwrite( "b", &RGB32f::b )    
        .def_readwrite( "g", &RGB32f::g )    
        .def_readwrite( "r", &RGB32f::r )    
        .def_readwrite( "c", &RGB32f::c );

}
