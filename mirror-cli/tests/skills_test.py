"""Tests for `SKILL.md` discovery and loading."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from mirror import project, skills


@pytest.fixture(autouse=True)
def _isolated_global_dir(
  tmp_path_factory: pytest.TempPathFactory,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Point the global dir at an empty path so tests don't read ~/.mirror."""
  empty = tmp_path_factory.mktemp('no_global')
  monkeypatch.setattr(project, 'GLOBAL_DIR', empty / 'missing')


def _make_repo(root: pathlib.Path) -> pathlib.Path:
  root.mkdir(parents=True, exist_ok=True)
  (root / '.git').mkdir()
  return root


def _write_skill(
  root: pathlib.Path,
  name: str,
  frontmatter: str,
  body: str = 'Do the thing.',
  marker: pathlib.Path = skills.SKILL_ROOT_DIRS[0],
) -> pathlib.Path:
  skill_dir = root / marker / name
  skill_dir.mkdir(parents=True)
  path = skill_dir / skills.SKILL_FILENAME
  path.write_text(f'---\n{textwrap.dedent(frontmatter).strip()}\n---\n{body}')
  return path


def test_loads_name_and_description_from_frontmatter(
  tmp_path: pathlib.Path,
) -> None:
  _make_repo(tmp_path)
  _write_skill(tmp_path, 'pdf', 'name: pdf\ndescription: Work with PDF files.')
  loaded = skills.load_skills(tmp_path)
  assert len(loaded) == 1
  assert loaded[0].name == 'pdf'
  assert loaded[0].description == 'Work with PDF files.'
  assert loaded[0].body == 'Do the thing.'


def test_invocation_content_wraps_skill_as_contextual_user_fragment() -> None:
  skill = skills.Skill(
    'review&ship',
    'Review.',
    'Follow these instructions.',
    pathlib.Path('/skills/review&ship/SKILL.md'),
  )

  assert skill.invocation_content() == (
    '<skill>\n<name>review&amp;ship</name>\n'
    '<path>/skills/review&amp;ship/SKILL.md</path>\n'
    'Follow these instructions.\n</skill>'
  )


def test_name_defaults_to_directory_when_absent(
  tmp_path: pathlib.Path,
) -> None:
  _make_repo(tmp_path)
  _write_skill(tmp_path, 'commit-helper', 'description: Write commits.')
  loaded = skills.load_skills(tmp_path)
  assert loaded[0].name == 'commit-helper'


def test_folded_block_scalar_description(tmp_path: pathlib.Path) -> None:
  _make_repo(tmp_path)
  _write_skill(
    tmp_path,
    'dataplane',
    """
    name: dataplane
    description: >
      Reflection's dataplane platform for GCP batch processing
      (Dataproc Serverless Spark and Cloud Dataflow jobs).
    """,
  )
  loaded = skills.load_skills(tmp_path)
  assert loaded[0].description == (
    "Reflection's dataplane platform for GCP batch processing "
    '(Dataproc Serverless Spark and Cloud Dataflow jobs).'
  )


def test_skill_without_description_is_skipped(tmp_path: pathlib.Path) -> None:
  _make_repo(tmp_path)
  _write_skill(tmp_path, 'broken', 'name: broken')
  assert skills.load_skills(tmp_path) == []


def test_full_yaml_frontmatter_and_typed_fields(tmp_path: pathlib.Path) -> None:
  _make_repo(tmp_path)
  _write_skill(
    tmp_path,
    'deploy',
    """
    name: deploy
    description: "Deploy safely: with checks"
    disable-model-invocation: true
    metadata:
      owner: platform
    """,
  )

  loaded = skills.load_skills(tmp_path)[0]
  assert loaded.description == 'Deploy safely: with checks'
  assert loaded.disable_model_invocation


def test_non_string_description_is_skipped(tmp_path: pathlib.Path) -> None:
  _make_repo(tmp_path)
  _write_skill(tmp_path, 'broken', 'description: true')

  assert skills.load_skills(tmp_path) == []


def test_frontmatter_requires_exact_delimiter_lines(
  tmp_path: pathlib.Path,
) -> None:
  _make_repo(tmp_path)
  skill_dir = tmp_path / skills.SKILL_ROOT_DIRS[0] / 'broken'
  skill_dir.mkdir(parents=True)
  (skill_dir / skills.SKILL_FILENAME).write_text(
    '---not-frontmatter\ndescription: Should not load.\n---\nbody'
  )

  assert skills.load_skills(tmp_path) == []


def test_repairs_unquoted_colons_in_plain_scalars(
  tmp_path: pathlib.Path,
) -> None:
  _make_repo(tmp_path)
  _write_skill(
    tmp_path,
    'deploy',
    """
    name: deploy
    description: Deploy to AWS: ECS
    argument-hint: <duration: e.g. 7d>
    """,
  )

  assert skills.load_skills(tmp_path)[0].description == 'Deploy to AWS: ECS'


def test_claude_skills_root_is_discovered(tmp_path: pathlib.Path) -> None:
  _make_repo(tmp_path)
  _write_skill(
    tmp_path,
    'legacy',
    'description: A Claude-style skill.',
    marker=pathlib.Path('.claude') / 'skills',
  )
  loaded = skills.load_skills(tmp_path)
  assert [skill.name for skill in loaded] == ['legacy']


def test_discovers_skills_up_to_project_root(tmp_path: pathlib.Path) -> None:
  repo = _make_repo(tmp_path / 'repo')
  _write_skill(repo, 'root-skill', 'description: At the root.')
  nested = repo / 'sub' / 'deep'
  nested.mkdir(parents=True)
  _write_skill(nested, 'deep-skill', 'description: Deep in the tree.')

  loaded = skills.load_skills(nested)
  assert {skill.name for skill in loaded} == {'root-skill', 'deep-skill'}


def test_recursively_discovers_nested_skill_directories(
  tmp_path: pathlib.Path,
) -> None:
  _make_repo(tmp_path)
  skill_dir = (
    tmp_path / skills.SKILL_ROOT_DIRS[0] / 'collection' / 'nested-skill'
  )
  skill_dir.mkdir(parents=True)
  (skill_dir / skills.SKILL_FILENAME).write_text(
    '---\nname: nested-skill\ndescription: Nested.\n---\nbody'
  )

  assert [skill.name for skill in skills.load_skills(tmp_path)] == [
    'nested-skill'
  ]


def test_discovery_does_not_descend_into_loaded_skill(
  tmp_path: pathlib.Path,
) -> None:
  _make_repo(tmp_path)
  outer = tmp_path / skills.SKILL_ROOT_DIRS[0] / 'outer'
  nested = outer / 'references' / 'nested'
  nested.mkdir(parents=True)
  (outer / skills.SKILL_FILENAME).write_text(
    '---\nname: outer\ndescription: Outer.\n---\nbody'
  )
  (nested / skills.SKILL_FILENAME).write_text(
    '---\nname: nested\ndescription: Nested.\n---\nbody'
  )

  assert [skill.name for skill in skills.load_skills(tmp_path)] == ['outer']


def test_discovery_skips_hidden_directories_and_binary_skill_files(
  tmp_path: pathlib.Path,
) -> None:
  _make_repo(tmp_path)
  root = tmp_path / skills.SKILL_ROOT_DIRS[0]
  binary = root / 'binary'
  hidden = root / '.git' / 'hidden'
  binary.mkdir(parents=True)
  hidden.mkdir(parents=True)
  (binary / skills.SKILL_FILENAME).write_bytes(b'\xff\xfe\x00')
  (hidden / skills.SKILL_FILENAME).write_text(
    '---\nname: hidden\ndescription: Hidden.\n---\nbody'
  )
  _write_skill(tmp_path, 'visible', 'description: Visible.')

  assert [skill.name for skill in skills.load_skills(tmp_path)] == ['visible']


def test_canonical_paths_deduplicate_symlinked_skill_roots(
  tmp_path: pathlib.Path,
) -> None:
  _make_repo(tmp_path)
  expected = _write_skill(tmp_path, 'pdf', 'description: Work with PDFs.')
  claude_dir = tmp_path / '.claude'
  claude_dir.mkdir()
  (claude_dir / 'skills').symlink_to('../.agents/skills')

  loaded = skills.load_skills(tmp_path)

  assert len(loaded) == 1
  assert loaded[0].path == expected


def test_discovery_ignores_child_symlink_outside_skill_root(
  tmp_path: pathlib.Path,
) -> None:
  _make_repo(tmp_path)
  external = tmp_path / 'external' / 'leaked'
  external.mkdir(parents=True)
  (external / skills.SKILL_FILENAME).write_text(
    '---\nname: leaked\ndescription: External.\n---\nsecret'
  )
  root = tmp_path / skills.SKILL_ROOT_DIRS[0]
  root.mkdir(parents=True)
  (root / 'linked').symlink_to(external, target_is_directory=True)

  assert skills.load_skills(tmp_path) == []


def test_invalid_skill_name_is_skipped(tmp_path: pathlib.Path) -> None:
  _make_repo(tmp_path)
  _write_skill(
    tmp_path,
    'invalid',
    'name: "not invokable"\ndescription: Invalid name.',
  )

  assert skills.load_skills(tmp_path) == []


def test_nearest_skill_overrides_shallower(tmp_path: pathlib.Path) -> None:
  repo = _make_repo(tmp_path / 'repo')
  _write_skill(repo, 'pdf', 'name: pdf\ndescription: Root PDF skill.')
  nested = repo / 'sub'
  nested.mkdir()
  _write_skill(nested, 'pdf', 'name: pdf\ndescription: Nested PDF skill.')

  loaded = skills.load_skills(nested)
  assert len(loaded) == 1
  assert loaded[0].description == 'Nested PDF skill.'


def test_project_skill_overrides_global(tmp_path: pathlib.Path) -> None:
  global_root = tmp_path / 'global'
  global_dir = global_root / 'pdf'
  global_dir.mkdir(parents=True)
  (global_dir / skills.SKILL_FILENAME).write_text(
    '---\nname: pdf\ndescription: Global PDF skill.\n---\nglobal body'
  )
  repo = _make_repo(tmp_path / 'repo')
  _write_skill(repo, 'pdf', 'name: pdf\ndescription: Project PDF skill.')

  loaded = skills.load_skills(repo, global_dir=global_root)
  assert len(loaded) == 1
  assert loaded[0].description == 'Project PDF skill.'


def test_include_global_false_skips_global(tmp_path: pathlib.Path) -> None:
  global_root = tmp_path / 'global'
  global_dir = global_root / 'pdf'
  global_dir.mkdir(parents=True)
  (global_dir / skills.SKILL_FILENAME).write_text(
    '---\nname: pdf\ndescription: Global.\n---\nbody'
  )
  repo = _make_repo(tmp_path / 'repo')
  assert (
    skills.load_skills(repo, global_dir=global_root, include_global=False) == []
  )


def test_disable_model_invocation_frontmatter(tmp_path: pathlib.Path) -> None:
  _make_repo(tmp_path)
  _write_skill(
    tmp_path,
    'manual',
    'description: Manual only.\ndisable-model-invocation: true',
  )

  assert skills.load_skills(tmp_path)[0].disable_model_invocation


def test_mentioned_skill_names_returns_known_unique_mentions() -> None:
  available = [
    skills.Skill('pdf', 'PDFs.', 'body', pathlib.Path('/pdf/SKILL.md')),
    skills.Skill('prs', 'PRs.', 'body', pathlib.Path('/prs/SKILL.md')),
  ]

  assert skills.mentioned_skill_names(
    'Use $pdf then $prs and $pdf again; ignore $HOME and $unknown',
    available,
  ) == ['pdf', 'prs']


def test_mentioned_skill_names_requires_exact_tokens() -> None:
  available = [
    skills.Skill('prs', 'PRs.', 'body', pathlib.Path('/prs/SKILL.md'))
  ]

  assert (
    skills.mentioned_skill_names('price$prs $prs-extra $prs, ($prs)', available)
    == []
  )
