/****************************************************************************
** Meta object code from reading C++ file 'managesocket.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../vrrenderer/managesocket.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'managesocket.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ManageSocket[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      14,   13,   13,   13, 0x0a,
      28,   13,   13,   13, 0x0a,
      55,   13,   13,   13, 0x0a,
      78,   13,   13,   13, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_ManageSocket[] = {
    "ManageSocket\0\0onreadyRead()\0"
    "download(QListWidgetItem*)\0"
    "load(QListWidgetItem*)\0onMessageConnect()\0"
};

void ManageSocket::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        ManageSocket *_t = static_cast<ManageSocket *>(_o);
        switch (_id) {
        case 0: _t->onreadyRead(); break;
        case 1: _t->download((*reinterpret_cast< QListWidgetItem*(*)>(_a[1]))); break;
        case 2: _t->load((*reinterpret_cast< QListWidgetItem*(*)>(_a[1]))); break;
        case 3: _t->onMessageConnect(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData ManageSocket::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject ManageSocket::staticMetaObject = {
    { &QTcpSocket::staticMetaObject, qt_meta_stringdata_ManageSocket,
      qt_meta_data_ManageSocket, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &ManageSocket::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *ManageSocket::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *ManageSocket::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ManageSocket))
        return static_cast<void*>(const_cast< ManageSocket*>(this));
    return QTcpSocket::qt_metacast(_clname);
}

int ManageSocket::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QTcpSocket::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
