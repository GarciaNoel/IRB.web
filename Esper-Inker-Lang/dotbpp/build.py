#!/usr/bin/python3

## LINUX ONLY

## Build Flags
#
# --windows [default:linux] - compile target for windows
# --gdb [default:disabled] - enables cmd debugger
# --gen-main [default:disabled] - generates c++ from python
#
##


import os, sys, subprocess

## SETUP / INSTALL ##

if "--windows" in sys.argv:
    os.system("rm /tmp/*.o /tmp/*.exe")

    ## https://stackoverflow.com/questions/43864159/mutex-is-not-a-member-of-std-in-mingw-5-3-0
    ## TODO, not use std::mutex? seems like the only issue using win32 instead os posix
    # CC  = 'i686-w64-mingw32-g++-win32'
    # C   = 'i686-w64-mingw32-gcc-win32'

    CC = "i686-w64-mingw32-g++-posix"
    C = "i686-w64-mingw32-gcc-posix"

    if not os.path.isfile(os.path.join("/usr/bin/", CC)):
        cmd = "sudo apt-get install mingw-w64 gcc-multilib g++-multilib"
        subprocess.check_call(cmd.split())

else:
    CC = "g++"
    C = "gcc"


srcdir = os.path.abspath("./Source")
assert os.path.isdir(srcdir)

includes = [
    "-I" + srcdir,
    # "-I/usr/include/freetype2",
]


def fake_includes():
    if os.path.isdir("/tmp/fake"):
        return
    os.system("mkdir /tmp/fake/")
    # os.system("cp -Rv /usr/include/GL /tmp/fake/.")
    # os.system("cp -Rv /usr/include/GLFW /tmp/fake/.")
    # os.system("cp -Rv /usr/include/glm /tmp/fake/.")
    # os.system("cp -Rv /usr/include/assimp /tmp/fake/.")
    # os.system("cp -Rv /usr/include/boost /tmp/fake/.")
    # os.system("cp -Rv /usr/include/AL /tmp/fake/.")


if "--windows" in sys.argv:
    # includes += ['-I/usr/include']
    # includes += ["-lopengl32", "-I/tmp/fake"]
    fake_includes()

hacks = [
    # "-I/usr/include/bullet",  ## ex
]

libs = [
    # "-lGL",
    # "-lGLU",
    # "-lGLEW",
    # "-lglfw",
    # "-lopenal",
    #'-lbullet',
    # "-lfreetype",
    # "-lBulletDynamics",
    # "-lBulletCollision",
    # "-lLinearMath",
    # "-lassimp",
    #'-lstdc',
    # "-lm",
    # "-lc",
    # "-lstdc++",
]

bpp_so = "/tmp/bpp.so"

## BUILD CODE ##

gen_main = "--gen-main" in sys.argv
test_main = "--test-main" in sys.argv


def testmain():
    o = []

    BPLATE = """
    int* data = new int[20];
    for(int i = 0; i < 20; i++){
        data[i] = i;
        std::cout << std::to_string(data[i]) << std::endl;
    }
    void* ret = dae.blockingProcess(TestFunc(20), (void*)data);
    int* retList = (int*)ret;
    for(int i = 0; i < 20; i++){
        std::cout << std::to_string(retList[i]) << std::endl;
    }
    """

    o.extend(
        [
            '#include "Run.h"',
            '#include "Daemon.h"',
            "int main(int argc, char **argv) {",
            BPLATE,
        ]
    )

    o.append("}")
    o = "\n".join(o)
    return o


def genmain():
    o = []

    BPLATE = """
    return run();
    """

    o.extend(
        [
            '#include "Run.h"',
            "int main(int argc, char **argv) {",
            BPLATE,
        ]
    )

    o.append("}")
    o = "\n".join(o)
    return o


def build():
    cpps = []
    obfiles = []

    for file in os.listdir(srcdir):

        if file == "Main.cpp":
            if gen_main:
                open("/tmp/gen.oe.main.cpp", "wb").write(genmain().encode("utf-8"))
                file = "/tmp/gen.oe.main.cpp"
                ofile = "%s.o" % file
                obfiles.append(ofile)
                # if os.path.isfile(ofile):
                #    continue
                cpps.append(file)
                cmd = [
                    #'g++',
                    CC,
                    "-std=c++20",
                    "-c",  ## do not call the linker
                    "-fPIC",  ## position indepenent code
                    "-o",
                    ofile,
                    os.path.join(srcdir, file),
                ]
                cmd += libs
                cmd += includes
                cmd += hacks
                print(cmd)
                subprocess.check_call(cmd)
            elif test_main:
                open("/tmp/gen.oe.main.cpp", "wb").write(testmain().encode("utf-8"))
                file = "/tmp/gen.oe.main.cpp"
                ofile = "%s.o" % file
                obfiles.append(ofile)
                # if os.path.isfile(ofile):
                #    continue
                cpps.append(file)
                cmd = [
                    #'g++',
                    CC,
                    "-std=c++20",
                    "-c",  ## do not call the linker
                    "-fPIC",  ## position indepenent code
                    "-o",
                    ofile,
                    os.path.join(srcdir, file),
                ]
                cmd += libs
                cmd += includes
                cmd += hacks
                print(cmd)
                subprocess.check_call(cmd)
            else:
                ofile = "/tmp/%s.o" % file
                obfiles.append(ofile)
                # if os.path.isfile(ofile):
                #    continue
                cpps.append(file)
                cmd = [
                    #'g++',
                    CC,
                    "-std=c++20",
                    "-c",  ## do not call the linker
                    "-fPIC",  ## position indepenent code
                    "-o",
                    ofile,
                    os.path.join(srcdir, file),
                ]
                cmd += libs
                cmd += includes
                cmd += hacks
                print(cmd)
                subprocess.check_call(cmd)

        elif file.endswith(".c"):
            ## this is just for drwave
            ofile = "/tmp/%s.o" % file
            obfiles.append(ofile)
            # if os.path.isfile(ofile):
            #    continue
            cpps.append(file)
            cmd = [
                #'gcc',
                C,
                "-c",  ## do not call the linker
                "-fPIC",  ## position indepenent code
                "-o",
                ofile,
                os.path.join(srcdir, file),
            ]
            print(cmd)
            subprocess.check_call(cmd)

        elif file.endswith(".cpp"):
            ofile = "/tmp/%s.o" % file
            obfiles.append(ofile)
            # if os.path.isfile(ofile):
            #    continue
            cpps.append(file)
            cmd = [
                #'g++',
                CC,
                "-std=c++20",
                "-c",  ## do not call the linker
                "-fPIC",  ## position indepenent code
                "-o",
                ofile,
                os.path.join(srcdir, file),
            ]
            cmd += libs
            cmd += includes
            cmd += hacks
            print(cmd)
            subprocess.check_call(cmd)

    os.system("ls -lh /tmp/*.o")

    cmd = (
        [
            #'ld',
            "g++",
            "-shared",
            "-o",
            "/tmp/bpp.so",
        ]
        + obfiles
        + libs
    )

    print(cmd)
    subprocess.check_call(cmd)

    exe = "/tmp/bpp"
    if "--windows" in sys.argv:
        exe += ".exe"
    cmd = [
        #'g++',
        CC,
        "-o",
        exe,
    ]
    if "--windows" in sys.argv:
        cmd += "-static-libgcc -static-libstdc++ -static".split()
    cmd += obfiles + libs

    print(cmd)

    subprocess.check_call(cmd)


## BUILDING / RUNNING ##

build()

if "--windows" in sys.argv:
    cmd = ["/tmp/bpp.exe"]
elif "--gdb" in sys.argv:
    cmd = ["gdb", "/tmp/bpp"]
else:
    cmd = ["/tmp/bpp"]

print(cmd)

import ctypes

dll = ctypes.CDLL(bpp_so)
print(dll)

print(dll.main)
dll.main()
