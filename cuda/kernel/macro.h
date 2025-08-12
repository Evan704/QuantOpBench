#pragma once
#define GET_INT4(pointer) (*(reinterpret_cast<int4*>(pointer)))
#define GET_INT2(pointer) (*(reinterpret_cast<int2*>(pointer)))