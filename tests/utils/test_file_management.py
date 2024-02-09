from ml.utils.file_management import check_savefile_integrity


class TestIntegrity:
    def test_correct_file(self):
        file1 = 'gen.pt'
        file2 = 'disc.pth'
        assert check_savefile_integrity(file1)
        assert check_savefile_integrity(file2)

    def test_not_correct_file(self):
        file1 = 'gen.pts'
        file2 = 'disc.txt'
        assert not check_savefile_integrity(file1)
        assert not check_savefile_integrity(file2)
