# CMakeList.txt : CMake project for GrandMassive, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.15)

# Add source to this project's executable.
add_executable (GrandMassive "GrandMassive.cpp" "GrandMassive.h" "uci.h" "Chess.h" "Search.h" "Eval.h" "TT.h" "Book.h" "BookData.h" "Nnue.h" "NetData.h" "Endgames.h")

# GrandMassive_tune: the same engine with SP_TUNE=1: every search parameter
# (SP_PARAM in Search.h) is a UCI spin option, for tools/spsa.cpp
# (tools/spsa_tune.bat) and tools/order_stats.bat. Not for play: slower.
add_executable (GrandMassive_tune "GrandMassive.cpp" "GrandMassive.h" "uci.h" "Chess.h" "Search.h" "Eval.h" "TT.h" "Book.h" "BookData.h" "Nnue.h" "NetData.h" "Endgames.h")
target_compile_definitions(GrandMassive_tune PRIVATE SP_TUNE=1)

# Test engines for other nets / layouts: built only when tools/train_net.bat has
# made the net (rerun the CMake configure after training). Each embeds
# tools/data/NetData_new_<variant>.h instead of NetData.h, with the matching
# layout, for SPRT against GrandMassive.exe.
set(GM_ENGINES GrandMassive GrandMassive_tune)
set(GM_NET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tools/data")
# A new net file there reruns the configure (the folder's time stamp changes).
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GM_NET_DIR}")
# "prev": the previous net (NetData_prev.h, the engine's own layout) built with
# today's search, the base for a new-net SPRT: GrandMassive.exe vs
# GrandMassive_prev.exe differ only in the net.
foreach(variant kb4 h512 h1024 prev)
    if(variant STREQUAL "prev")
        set(net "${GM_NET_DIR}/NetData_prev.h")
    else()
        set(net "${GM_NET_DIR}/NetData_new_${variant}.h")
    endif()
    if(EXISTS "${net}")
        add_executable (GrandMassive_${variant} "GrandMassive.cpp" "GrandMassive.h" "uci.h" "Chess.h" "Search.h" "Eval.h" "TT.h" "Book.h" "BookData.h" "Nnue.h" "Endgames.h")
        target_compile_definitions(GrandMassive_${variant} PRIVATE "NNUE_NET_FILE=\"${net}\"")
        # The layout comes from the net file itself (its static_assert line),
        # so e.g. "prev" builds whatever layout the previous net had.
        file(STRINGS "${net}" layout REGEX "static_assert\\(Hidden == " LIMIT_COUNT 1)
        if(layout MATCHES "Hidden == ([0-9]+) && OutBuckets == ([0-9]+) && KingBuckets == ([0-9]+)")
            target_compile_definitions(GrandMassive_${variant} PRIVATE NNUE_HIDDEN=${CMAKE_MATCH_1}
                                       NNUE_OUT_BUCKETS=${CMAKE_MATCH_2} NNUE_KING_BUCKETS=${CMAKE_MATCH_3})
        else()
            message(WARNING "${net}: no layout line found; using the engine's default layout")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${net}")
        list(APPEND GM_ENGINES GrandMassive_${variant})
        message(STATUS "test engine GrandMassive_${variant}: ${net}")
    endif()
endforeach()

foreach(engine ${GM_ENGINES})
target_compile_features(${engine} PRIVATE cxx_std_20)

# Windows: the icon (GrandMassive.ico, via GrandMassive.rc) for every engine exe.
if(WIN32)
    target_sources(${engine} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/GrandMassive.rc")
endif()

# Evaluation: ON = NNUE (Nnue.h / NetData.h), OFF = the hand-written eval.
option(GM_NNUE "Use the NNUE evaluation (OFF: hand-written eval)" ON)
if(GM_NNUE)
    target_compile_definitions(${engine} PRIVATE EV_NNUE=1)
else()
    target_compile_definitions(${engine} PRIVATE EV_NNUE=0)
endif()

# True for optimized builds (Release and RelWithDebInfo)
set(OPTIMIZED "$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>")

if(MSVC)  # MSVC and clang-cl
    target_compile_options(${engine} PRIVATE
        /permissive-            # strict standards conformance
        /Zc:__cplusplus         # report the real __cplusplus value
        /W4
        "$<${OPTIMIZED}:/O2;/Ob2;/Oi;/Ot;/GS-;/Gw;/DNDEBUG>"
    )
    if(GM_NATIVE)
        target_compile_options(${engine} PRIVATE /arch:AVX2)
    endif()
    target_link_options(${engine} PRIVATE
        "$<${OPTIMIZED}:/OPT:REF;/OPT:ICF>"
    )
else()    # GCC / Clang
    target_compile_options(${engine} PRIVATE
        -Wall -Wextra
        "$<${OPTIMIZED}:-O3;-DNDEBUG>"
    )
    if(GM_NATIVE)
        target_compile_options(${engine} PRIVATE -march=native)
    endif()
endif()

# Lets code check "#if GM_NATIVE" before using pext/pdep, AVX2 intrinsics, etc.
if(GM_NATIVE)
    target_compile_definitions(${engine} PRIVATE GM_NATIVE=1)
endif()

# Link-time optimization (/GL + /LTCG on MSVC, -flto elsewhere) for optimized builds
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_OK OUTPUT IPO_MSG LANGUAGES CXX)
if(IPO_OK)
    set_property(TARGET ${engine} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET ${engine} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
else()
    message(STATUS "LTO not supported: ${IPO_MSG}")
endif()
endforeach()

# Copy EPD test suites next to the executable so "testsuite wac.epd" works
# when the engine is run from its build folder.
configure_file(wac.epd ${CMAKE_CURRENT_BINARY_DIR}/wac.epd COPYONLY)

# TODO: Add tests and install targets if needed.
