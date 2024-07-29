/****************************************************************************
** Meta object code from reading C++ file 'v3dr_gl_vr.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../vrrenderer/v3dr_gl_vr.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'v3dr_gl_vr.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_CMainApplication[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      23,   18,   17,   17, 0x0a,
      42,   17,   17,   17, 0x0a,
      59,   17,   17,   17, 0x0a,
      79,   17,   74,   17, 0x0a,
      98,   17,   17,   17, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_CMainApplication[] = {
    "CMainApplication\0\0show\0ImageDisplay(bool)\0"
    "onTimerTimeout()\0timerTimeout()\0bool\0"
    "startBCIparadigm()\0stopBCIparadigm()\0"
};

void CMainApplication::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        CMainApplication *_t = static_cast<CMainApplication *>(_o);
        switch (_id) {
        case 0: _t->ImageDisplay((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->onTimerTimeout(); break;
        case 2: _t->timerTimeout(); break;
        case 3: { bool _r = _t->startBCIparadigm();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 4: _t->stopBCIparadigm(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData CMainApplication::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject CMainApplication::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_CMainApplication,
      qt_meta_data_CMainApplication, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CMainApplication::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CMainApplication::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CMainApplication::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CMainApplication))
        return static_cast<void*>(const_cast< CMainApplication*>(this));
    return QObject::qt_metacast(_clname);
}

int CMainApplication::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
