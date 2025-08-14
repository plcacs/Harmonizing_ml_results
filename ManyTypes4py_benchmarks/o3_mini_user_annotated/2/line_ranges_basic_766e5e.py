# flags: --line-ranges=5-6
# NOTE: If you need to modify this file, pay special attention to the --line-ranges=
# flag above as it's formatting specifically these lines.
def foo1(
    parameter_1: Any,
    parameter_2: Any,
    parameter_3: Any,
    parameter_4: Any,
    parameter_5: Any,
    parameter_6: Any,
    parameter_7: Any,
) -> None: pass
def foo2(
    parameter_1: Any,
    parameter_2: Any,
    parameter_3: Any,
    parameter_4: Any,
    parameter_5: Any,
    parameter_6: Any,
    parameter_7: Any,
) -> None:
    pass


def foo3(
    parameter_1: Any,
    parameter_2: Any,
    parameter_3: Any,
    parameter_4: Any,
    parameter_5: Any,
    parameter_6: Any,
    parameter_7: Any,
) -> None:
    pass


def foo4(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None: pass

# Adding some unformated code covering a wide range of syntaxes.

if True:
      # Incorrectly indented prefix comments.
  pass

import   typing
from  typing   import   (
      Any  ,
   )
class   MyClass(  object):     # Trailing comment with extra leading space.
        #NOTE: The following indentation is incorrect:
      @decor( 1  *  3 )
      def  my_func(self, arg: Any) -> None:
                pass

try:                                       # Trailing comment with extra leading space.
    for   i   in   range(10):              # Trailing comment with extra leading space.
        while    condition:
            if   something:
                then_something(  )
            elif    something_else:
                then_something_else(  )
except  ValueError  as  e:
    unformatted(  )
finally:
    unformatted(  )

async  def  test_async_unformatted(  ) -> None:    # Trailing comment with extra leading space.
    async  for  i  in some_iter(  unformatted  ):    # Trailing comment with extra leading space.
        await  asyncio.sleep( 1 )
        async  with  some_context(  unformatted  ):
            print(  "unformatted"  )