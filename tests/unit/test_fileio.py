from __future__ import annotations

from pathlib import Path
import subprocess
import zipfile

import numpy as np
from scipy.io import loadmat, whosmat

from pyisetcam import (
    Camera,
    Scene,
    camera_create,
    display_create,
    ieImageType,
    ieDNGRead,
    ieDNGSimpleInfo,
    ieSCP,
    ieSaveSpectralFile,
    ieStruct2XML,
    ieTempfile,
    ieVarInFile,
    ieWebGet,
    ieXML2struct,
    ieXL2ColorFilter,
    ie_dng_read,
    ie_dng_simple_info,
    ie_image_type,
    ie_scp,
    ie_save_spectral_file,
    ie_struct2xml,
    ie_tempfile,
    ie_var_in_file,
    ie_web_get,
    ie_xml2struct,
    ie_xl2_color_filter,
    pathToLinux,
    path_to_linux,
    scene_create,
    scene_from_file,
    sensorDNGRead,
    sensor_crop,
    sensor_dng_read,
    sensor_get,
    session_create,
    session_get_selected,
    vcExportObject,
    vcImportObject,
    vcLoadObject,
    vcReadImage,
    vcReadSpectra,
    vcSaveObject,
    vcSaveMultiSpectralImage,
    vc_export_object,
    vc_import_object,
    vc_load_object,
    vc_read_image,
    vc_read_spectra,
    vc_save_object,
    vc_save_multispectral_image,
    struct2xml,
    xml2struct,
)


def test_vc_save_and_load_object_round_trip_scene(tmp_path, asset_store) -> None:
    scene = scene_create("uniform ee", 8, asset_store=asset_store)
    path = tmp_path / "scene_round_trip.mat"

    saved = vc_save_object(scene, path)
    loaded, full_name = vc_load_object("scene", saved)

    assert full_name == str(path)
    assert path.exists()
    assert isinstance(loaded, Scene)
    assert loaded.name == "scene_round_trip"
    assert loaded.type == "scene"
    assert np.allclose(np.asarray(loaded.fields["wave"], dtype=float), np.asarray(scene.fields["wave"], dtype=float))
    assert np.allclose(np.asarray(loaded.data["photons"], dtype=float), np.asarray(scene.data["photons"], dtype=float))
    assert vcSaveObject(scene, tmp_path / "scene_round_trip_alias.mat").endswith(".mat")


def test_vc_export_object_clear_data_flag_preserves_original(tmp_path, asset_store) -> None:
    scene = scene_create("uniform d65", 8, asset_store=asset_store)
    original_photons = np.asarray(scene.data["photons"], dtype=float).copy()
    path = tmp_path / "scene_export.mat"

    saved = vc_export_object(scene, path, clear_data_flag=True)
    loaded, _ = vc_load_object("scene", saved)

    assert np.allclose(np.asarray(scene.data["photons"], dtype=float), original_photons)
    assert loaded.data == {}
    assert vcExportObject(scene, tmp_path / "scene_export_alias.mat", clear_data_flag=False).endswith(".mat")


def test_vc_load_object_registers_in_session(tmp_path, asset_store) -> None:
    scene = scene_create("checkerboard", 4, 4, asset_store=asset_store)
    path = tmp_path / "session_scene.mat"
    session = session_create()

    vc_save_object(scene, path)
    slot, full_name = vc_load_object("scene", path, session=session)

    assert slot == 1
    assert full_name == str(path)
    loaded = session_get_selected(session, "scene")
    assert isinstance(loaded, Scene)
    assert loaded.name == "session_scene"
    assert np.allclose(np.asarray(loaded.data["photons"], dtype=float), np.asarray(scene.data["photons"], dtype=float))


def test_vc_save_and_load_camera_reconstructs_nested_objects(tmp_path, asset_store) -> None:
    camera = camera_create(asset_store=asset_store)
    path = tmp_path / "camera_round_trip.mat"

    vc_save_object(camera, path)
    loaded, _ = vcLoadObject("camera", path)

    assert isinstance(loaded, Camera)
    assert loaded.name == "camera_round_trip"
    assert loaded.fields["oi"].type == "opticalimage"
    assert loaded.fields["sensor"].type == "sensor"
    assert loaded.fields["ip"].type == "vcimage"
    assert loaded.fields["ip"].fields["display"].type == "display"


def test_ie_dng_read_supports_raw_rgb_and_simple_info(asset_store) -> None:
    dng_path = asset_store.resolve("data/images/rawcamera/MCC-centered.dng")

    raw, info = ie_dng_read(dng_path)
    rgb, _ = ie_dng_read(dng_path, "rgb", True)
    none_img, simple_info = ie_dng_read(dng_path, "only info", True, "simple info", True)

    assert raw is not None
    assert raw.shape == (3024, 4032)
    assert raw.dtype == np.uint16
    assert info["Make"] == "Google"
    assert info["Model"] == "Pixel 4a"
    assert info["Orientation"] == 6
    assert int(info["DigitalCamera"]["ISOSpeedRatings"]) == 64
    assert np.allclose(np.asarray(info["BlackLevel"], dtype=float), np.array([1023.0, 1023.0, 1022.0, 1022.0]))

    assert rgb is not None
    assert rgb.shape == (504, 672, 3)
    assert rgb.dtype == np.uint8

    assert none_img is None
    assert simple_info["isoSpeed"] == 64
    assert np.isclose(simple_info["exposureTime"], 35822828 / 1073741824)
    assert simple_info["orientation"] == 6
    assert np.allclose(np.asarray(simple_info["blackLevel"], dtype=float), np.array([1023.0, 1023.0, 1022.0, 1022.0]))

    alias_img, alias_info = ieDNGRead(dng_path, "simple info", True)
    direct_simple = ieDNGSimpleInfo(info)
    assert alias_img is not None
    assert alias_info["isoSpeed"] == direct_simple["isoSpeed"] == ie_dng_simple_info(info)["isoSpeed"]
    assert np.isclose(alias_info["exposureTime"], direct_simple["exposureTime"])
    assert alias_info["orientation"] == direct_simple["orientation"]
    assert np.allclose(alias_info["blackLevel"], direct_simple["blackLevel"])


def test_sensor_dng_read_supports_imx363_raw_tutorial_flow(asset_store) -> None:
    dng_path = asset_store.resolve("data/images/rawcamera/MCC-centered.dng")

    sensor, info = sensor_dng_read(dng_path, asset_store=asset_store)
    cropped, simple_info = sensor_dng_read(
        dng_path,
        "full info",
        False,
        "crop",
        [500, 1000, 2500, 2500],
        asset_store=asset_store,
    )
    fractional, _ = sensorDNGRead(dng_path, "crop", 0.4, asset_store=asset_store)

    assert sensor.name == str(dng_path)
    assert sensor_get(sensor, "size") == (3024, 4032)
    assert np.array_equal(sensor_get(sensor, "pattern"), np.array([[2, 1], [3, 2]], dtype=int))
    assert sensor_get(sensor, "black level") == 1023
    assert np.isclose(sensor_get(sensor, "exp time"), 35822828 / 1073741824)
    dv = np.asarray(sensor_get(sensor, "digital values"), dtype=float)
    assert dv.shape == (3024, 4032)
    assert float(dv.min()) >= 1023.0

    manual_crop = sensor_crop(sensor, [1000, 500, 2500, 2500])
    assert sensor_get(cropped, "size") == sensor_get(manual_crop, "size")
    assert np.array_equal(sensor_get(cropped, "metadata crop"), sensor_get(manual_crop, "metadata crop"))
    assert np.array_equal(sensor_get(cropped, "digital values"), sensor_get(manual_crop, "digital values"))

    assert isinstance(info["DigitalCamera"], dict)
    assert simple_info["isoSpeed"] == 64
    assert sensor_get(fractional, "size")[0] < sensor_get(sensor, "size")[0]
    assert sensor_get(fractional, "size")[1] < sensor_get(sensor, "size")[1]


def test_legacy_spectral_file_helpers_round_trip(tmp_path) -> None:
    wave = np.array([400.0, 500.0, 600.0], dtype=float)
    data = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], dtype=float)
    path = tmp_path / "spectra.mat"

    saved = ie_save_spectral_file(wave, data, "demo spectral data", path)
    read_data, read_wave, comment = vc_read_spectra(saved, [400.0, 450.0, 500.0, 550.0, 600.0])

    assert saved == str(path)
    assert path.exists()
    assert np.allclose(read_wave, np.array([400.0, 450.0, 500.0, 550.0, 600.0]))
    assert np.allclose(
        read_data,
        np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ],
            dtype=float,
        ),
    )
    assert comment == "demo spectral data"
    assert ieSaveSpectralFile(wave, data, "alias", tmp_path / "spectra_alias.mat").endswith(".mat")
    alias_data, alias_wave, alias_comment = vcReadSpectra(saved, wave)
    assert np.allclose(alias_data, data)
    assert np.allclose(alias_wave, wave)
    assert alias_comment == "demo spectral data"


def test_legacy_path_tempfile_and_image_type_helpers(tmp_path) -> None:
    full_name, temp_dir = ie_tempfile("mat")

    assert full_name.endswith(".mat")
    assert temp_dir == str(Path(full_name).parent)
    assert Path(full_name).name.startswith("ie_")
    assert not Path(full_name).exists()
    assert ieTempfile("txt")[0].endswith(".txt")

    rgb_name = tmp_path / "Fruit-hdrs.png"
    mono_name = tmp_path / "Monochrome" / "target.png"
    multi_name = tmp_path / "multispectral" / "Fruit-hdrs.mat"
    mono_name.parent.mkdir()
    multi_name.parent.mkdir()

    assert ie_image_type(rgb_name) == "rgb"
    assert ieImageType(mono_name) == "monochrome"
    assert ie_image_type(multi_name) == "multispectral"
    assert path_to_linux(r"C:\Users\alice\iset\data") == "/Users/alice/iset/data"
    assert pathToLinux("/tmp/iset/data") == "/tmp/iset/data"


def test_ie_var_in_file_accepts_path_and_whos_listing(tmp_path) -> None:
    wave = np.array([400.0, 500.0, 600.0], dtype=float)
    data = np.eye(3, dtype=float)
    path = tmp_path / "vars.mat"
    ie_save_spectral_file(wave, data, "vars", path)

    assert ie_var_in_file(path, "data") is True
    assert ieVarInFile(path, "missing") is False
    assert ie_var_in_file(whosmat(path), "comment") is True


def test_ie_scp_builds_recursive_command_and_handles_missing_binary(tmp_path, monkeypatch, capsys) -> None:
    local_path = tmp_path / "download"

    def fake_run(command: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
        assert command == ["scp", "-r", "-q", "alice@example.com:/remote/data", str(local_path)]
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("pyisetcam.fileio.subprocess.run", fake_run)
    command, status = ie_scp("alice", "example.com", "/remote/data", local_path, "quiet", True)

    assert status == 0
    assert command == f"scp -r -q alice@example.com:/remote/data {local_path}"
    assert capsys.readouterr().out == ""

    def missing_run(command: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("scp")

    monkeypatch.setattr("pyisetcam.fileio.subprocess.run", missing_run)
    alias_command, alias_status = ieSCP("alice", "example.com", "/remote/data", local_path)

    assert alias_status == 127
    assert alias_command == f"scp -r alice@example.com:/remote/data {local_path}"
    assert "Error during remote secure copy" in capsys.readouterr().out


def test_vc_save_multispectral_image_and_import_object_wrappers(tmp_path, asset_store) -> None:
    wave = np.array([400.0, 500.0, 600.0], dtype=float)
    basis = {"wave": wave, "basis": np.eye(3, dtype=float)}
    basis_lights = np.array([[1.0, 0.5], [0.8, 0.4], [0.6, 0.3]], dtype=float)
    illuminant = {"wave": wave, "data": np.array([1.0, 2.0, 3.0], dtype=float)}
    mc_coef = np.ones((2, 2, 3), dtype=float)

    saved_ms = vc_save_multispectral_image(
        tmp_path,
        "example-hdrs.mat",
        mc_coef,
        basis,
        basis_lights,
        illuminant,
        "example comment",
        np.array([0.1, 0.2, 0.3], dtype=float),
    )
    loaded_ms = loadmat(saved_ms, squeeze_me=True, struct_as_record=False)

    assert Path(saved_ms).exists()
    assert loaded_ms["comment"] == "example comment"
    assert np.allclose(np.asarray(loaded_ms["basisLights"], dtype=float), basis_lights)
    assert vcSaveMultiSpectralImage(
        tmp_path,
        "example-alias-hdrs.mat",
        mc_coef,
        basis,
        basis_lights,
        illuminant,
        "alias comment",
        None,
    ).endswith(".mat")

    scene = scene_create("uniform ee", 8, asset_store=asset_store)
    object_path = tmp_path / "scene_import.mat"
    vc_save_object(scene, object_path)
    session = session_create()

    slot, full_name = vc_import_object("scene", object_path, session=session)

    assert slot == 1
    assert full_name == str(object_path)
    assert isinstance(session_get_selected(session, "scene"), Scene)
    alias_slot, alias_name = vcImportObject("scene", object_path, session=session_create())
    assert alias_slot == 1
    assert alias_name == str(object_path)


def test_ie_web_get_supports_list_browse_and_zip_download(tmp_path, monkeypatch) -> None:
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    archive_path = remote_dir / "payload.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/example.txt", "demo")

    monkeypatch.setattr(
        "pyisetcam.fileio._IE_WEB_GET_RESOURCES",
        {"testdeposit": remote_dir.as_uri()},
    )
    monkeypatch.setattr("pyisetcam.fileio._IE_WEB_GET_COLLECTIONS", {"testcollection": "https://example.com/collection"})

    listed = ie_web_get("list")
    browse_url = ieWebGet("browse", "testdeposit")
    local_dir, files = ie_web_get(
        "deposit name",
        "testdeposit",
        "deposit file",
        "payload.zip",
        "download dir",
        tmp_path / "downloaded",
        "confirm",
        False,
    )
    zip_only, zip_files = ieWebGet(
        "deposit name",
        "testdeposit",
        "deposit file",
        "payload.zip",
        "download dir",
        tmp_path / "zip-only",
        "unzip",
        False,
    )

    assert "testdeposit" in listed["deposits"]
    assert "testcollection" in listed["collections"]
    assert browse_url == remote_dir.as_uri()
    assert Path(local_dir).is_dir()
    assert len(files) == 1
    assert Path(files[0]).read_text() == "demo"
    assert Path(zip_only).is_file()
    assert zip_files == []


def test_ie_xl2_color_filter_reads_csv_and_saves_payloads(tmp_path) -> None:
    color_csv = tmp_path / "filters.csv"
    color_csv.write_text(
        "wavelength,rFilter,gFilter\n400,10,20\n500,30,40\n600,50,60\n",
        encoding="utf-8",
    )

    saved_filter, wavelength, data, comment = ie_xl2_color_filter(color_csv, tmp_path / "filters.mat")
    payload = loadmat(saved_filter, squeeze_me=True, struct_as_record=False)

    assert comment == "filters.csv"
    assert np.allclose(wavelength, np.array([400.0, 500.0, 600.0], dtype=float))
    assert np.allclose(data, np.array([[0.10, 0.20], [0.30, 0.40], [0.50, 0.60]], dtype=float))
    assert np.allclose(np.asarray(payload["data"], dtype=float), data)
    assert list(np.atleast_1d(payload["filterNames"]).tolist()) == ["rFilter", "gFilter"]

    spectral_csv = tmp_path / "spectral.csv"
    spectral_csv.write_text(
        "wavelength,a,b\n400,1,2\n500,3,4\n600,5,6\n",
        encoding="utf-8",
    )
    saved_spectral, _, spectral_data, _ = ieXL2ColorFilter(spectral_csv, tmp_path / "spectral.mat", "spectraldata")
    spectral_payload = loadmat(saved_spectral, squeeze_me=True, struct_as_record=False)

    assert np.allclose(spectral_data, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float))
    assert np.allclose(np.asarray(spectral_payload["data"], dtype=float), spectral_data)


def test_vc_read_image_matches_scene_from_file_and_multispectral_metadata(tmp_path, asset_store) -> None:
    rgb = np.array(
        [
            [[0.1, 0.2, 0.3], [0.7, 0.5, 0.2]],
            [[0.9, 0.4, 0.1], [0.3, 0.6, 0.8]],
        ],
        dtype=float,
    )
    display = display_create(asset_store=asset_store)
    photons, illuminant, basis, comment, mc_coef = vc_read_image(rgb, "rgb", display, asset_store=asset_store)
    scene = scene_from_file(rgb, "rgb", None, display, asset_store=asset_store)

    assert basis is None
    assert comment == ""
    assert mc_coef is None
    assert illuminant is not None
    assert np.allclose(photons, np.asarray(scene.data["photons"], dtype=float))
    assert np.allclose(np.asarray(illuminant["wave"], dtype=float), np.asarray(scene.fields["wave"], dtype=float))

    wave = np.array([400.0, 500.0, 600.0], dtype=float)
    basis_payload = {"wave": wave, "basis": np.eye(3, dtype=float)}
    illuminant_payload = {"wave": wave, "data": np.array([1.0, 2.0, 3.0], dtype=float)}
    mc_coef_payload = np.ones((2, 2, 3), dtype=float)
    multispectral_path = tmp_path / "basis_scene.mat"
    vc_save_multispectral_image(
        tmp_path,
        multispectral_path.name,
        mc_coef_payload,
        basis_payload,
        None,
        illuminant_payload,
        "basis comment",
        None,
    )

    photons_ms, illuminant_ms, basis_ms, comment_ms, mc_coef_ms = vcReadImage(
        multispectral_path,
        "multispectral",
        wave,
        asset_store=asset_store,
    )
    expected_scene = scene_from_file(multispectral_path, "multispectral", None, None, wave, asset_store=asset_store)

    assert illuminant_ms is not None
    assert basis_ms is not None
    assert comment_ms == "basis comment"
    assert np.allclose(photons_ms, np.asarray(expected_scene.data["photons"], dtype=float))
    assert np.allclose(np.asarray(basis_ms["wave"], dtype=float), wave)
    assert np.allclose(np.asarray(mc_coef_ms, dtype=float), mc_coef_payload)


def test_xml_helpers_round_trip_nested_payload_and_aliases(tmp_path) -> None:
    payload = {
        "Root_dash_node": {
            "Attributes": {"ns_colon_id": "17", "part_dot_name": "demo"},
            "Element": {"Text": "Some text"},
            "DifferentElement": [
                {"Attributes": {"attrib2": "2"}, "Text": "Some more text"},
                {"Attributes": {"attrib3": "2", "attrib4": "1"}, "Text": "Even more text"},
            ],
            "Nested_dot_child": {"Sub_dash_item": {"Text": "leaf"}},
        }
    }

    xml_text = ie_struct2xml(payload)
    parsed = ie_xml2struct(xml_text)

    assert "<?xml version='1.0' encoding='utf-8'?>" in xml_text
    assert "<Root-node" in xml_text
    assert 'ns:id="17"' in xml_text
    assert 'part.name="demo"' in xml_text
    assert parsed == payload
    assert ieXML2struct(xml_text) == payload
    assert xml2struct(xml_text) == payload
    assert struct2xml(payload).startswith("<?xml version='1.0' encoding='utf-8'?>")

    saved = ieStruct2XML(payload, tmp_path / "demo_payload")
    assert saved.endswith(".xml")
    assert Path(saved).exists()
    assert ie_xml2struct(tmp_path / "demo_payload") == payload


def test_xml_helpers_accept_file_input_and_attributes(tmp_path) -> None:
    xml_path = tmp_path / "simple.xml"
    xml_path.write_text(
        "<?xml version='1.0' encoding='utf-8'?><Root><Node attr='1'>value</Node></Root>",
        encoding="utf-8",
    )

    parsed = ie_xml2struct(xml_path)

    assert parsed == {"Root": {"Node": {"Attributes": {"attr": "1"}, "Text": "value"}}}
