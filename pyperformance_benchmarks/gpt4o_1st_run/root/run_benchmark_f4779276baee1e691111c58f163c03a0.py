'Test the performance of the Django template system.\n\nThis will have Django generate a 150x150-cell HTML table.\n'
import pyperf
import django.conf
from django.template import Context, Template
from typing import List
DEFAULT_SIZE: int = 100

def bench_django_template(runner: pyperf.Runner, size: int) -> None:
    template: Template = Template('<table>\n{% for row in table %}\n<tr>{% for col in row %}<td>{{ col|escape }}</td>{% endfor %}</tr>\n{% endfor %}\n</table>\n    ')
    table: List[range] = [range(size) for _ in range(size)]
    context: Context = Context({'table': table})
    runner.bench_func('django_template', template.render, context)

def prepare_cmd(runner: pyperf.Runner, cmd: pyperf.Runner.argparser) -> None:
    cmd.append(('--table-size=%s' % runner.args.table_size))

if (__name__ == '__main__'):
    django.conf.settings.configure(TEMPLATES=[{'BACKEND': 'django.template.backends.django.DjangoTemplates'}])
    django.setup()
    runner: pyperf.Runner = pyperf.Runner()
    cmd = runner.argparser
    cmd.add_argument('--table-size', type=int, default=DEFAULT_SIZE, help=('Size of the HTML table, height and width (default: %s)' % DEFAULT_SIZE))
    args = runner.parse_args()
    runner.metadata['description'] = 'Django template'
    runner.metadata['django_version'] = django.__version__
    runner.metadata['django_table_size'] = args.table_size
    bench_django_template(runner, args.table_size)
